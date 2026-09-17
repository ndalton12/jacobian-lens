"""Throttled log progress and an atomic, machine-readable progress snapshot."""

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from experiments.short_hop.common import parser, write_json

logger = logging.getLogger(__name__)


class Progress:
    """Overall percentage estimates work, using layer spans as fit weights.

    No ETA is asserted: backward cost varies by target and hardware. Checkpoints
    remain the source of truth for completed work, not this display snapshot.
    """

    def __init__(self, output_dir, jobs, *, interval=10):
        self.path = Path(output_dir) / "progress.json"
        self.jobs = {
            name: dict(method=method, total=total, done=0, weight=weight)
            for name, method, total, weight in jobs
        }
        self.interval = interval
        self.started = time.monotonic()
        self.last_write = self.last_log = 0
        self.current = "starting"
        self.detail = "Loading model and checking saved work"
        self.status = "running"
        self.flush(force=True)

    def update(self, job, done, *, detail="", force=False):
        changed = self.current != job
        self.current, self.detail = job, detail
        self.jobs[job]["done"] = min(self.jobs[job]["total"], max(0, done))
        self.flush(force=force or changed)

    def callback(self, method):
        def receive(event):
            job = f"{method}/target_{event['target_layer']:02d}"
            done = event["prompt_done"]
            detail = f"{done}/{event['prompt_total']} prompts saved"
            if event["event"] == "backward":
                done += event["pass_done"] / event["pass_total"]
                detail = (
                    f"prompt {event['prompt_done'] + 1}/{event['prompt_total']}, "
                    f"backward batch {event['pass_done']}/{event['pass_total']}"
                )
            self.update(
                job,
                done,
                detail=detail,
                force=event["event"] in ("resume", "prompt_done", "complete"),
            )

        return receive

    def finish(self, status="complete", detail=""):
        self.status, self.detail = status, detail
        self.flush(force=True)

    def flush(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 2:
            return
        methods = defaultdict(lambda: [0.0, 0.0])
        for job in self.jobs.values():
            methods[job["method"]][0] += job["done"] * job["weight"]
            methods[job["method"]][1] += job["total"] * job["weight"]
        total_done = sum(v[0] for v in methods.values())
        total_work = sum(v[1] for v in methods.values())
        percent = 100 * total_done / total_work if total_work else 0
        data = dict(
            status=self.status,
            pid=os.getpid(),
            updated_at=time.time(),
            elapsed_seconds_this_invocation=now - self.started,
            overall_percent=percent,
            percent_basis="estimated work; fit weighted by layer span",
            current=self.current,
            detail=self.detail,
            jobs=self.jobs,
            methods={
                k: dict(percent=100 * v[0] / v[1] if v[1] else 0)
                for k, v in methods.items()
            },
        )
        write_json(self.path, data)
        self.last_write = now
        if force or now - self.last_log >= self.interval:
            method = self.jobs.get(self.current, {}).get("method")
            method_percent = data["methods"].get(method, {}).get("percent", 0)
            logger.info(
                "Overall ~%.1f%% | %s %.1f%% | %s | %s",
                percent,
                method or self.status,
                method_percent,
                self.current,
                self.detail,
            )
            self.last_log = now


def show_progress(run_dir):
    path = Path(run_dir) / "progress.json"
    if not path.exists():
        print(f"{run_dir}: no progress snapshot yet")
        return
    data = json.loads(path.read_text())
    state = data["status"]
    if state == "running":
        try:
            os.kill(data["pid"], 0)
        except ProcessLookupError:
            state = "interrupted (runner process is gone; reissue run to resume)"
        except PermissionError:
            pass
    age = max(0, time.time() - data["updated_at"])
    print(
        f"{run_dir}: {state} | overall ~{data['overall_percent']:.1f}% | updated {age:.0f}s ago"
    )
    print(
        " | ".join(
            f"{name} {info['percent']:.1f}%" for name, info in data["methods"].items()
        )
    )
    print(f"{data['current']}: {data['detail']}")


def main():
    p = parser(__doc__)
    p.add_argument("--run-dir", default="runs/tjlens-a40")
    args = p.parse_args()
    show_progress(args.run_dir)


if __name__ == "__main__":
    main()
