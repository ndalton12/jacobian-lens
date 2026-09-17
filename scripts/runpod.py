"""Save RunPod SSH defaults, sync code/results, and manage detached runs.

Run with uv run --no-project python scripts/runpod.py --help (stdlib only).
This manages an existing pod over SSH; it never creates or terminates paid pods.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".runpod.json"
EXCLUDES = [
    ".git/",
    ".venv/",
    "__pycache__/",
    "*.egg-info/",
    ".pytest_cache/",
    ".ruff_cache/",
    "runs/",
    ".runpod*",
    ".env*",
    "*.pem",
    "*.key",
    ".ssh/",
    ".codex/",
    ".agents/",
]


def validate(config):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", config["host"]):
        raise ValueError("host must be an SSH hostname or IPv4 address")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", config["user"]):
        raise ValueError("invalid SSH username")
    if not 1 <= int(config["port"]) <= 65535:
        raise ValueError("port must be between 1 and 65535")
    remote = config["remote_dir"]
    if (
        not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote)
        or ".." in remote.split("/")
        or len(remote.strip("/").split("/")) < 2
    ):
        raise ValueError(
            "remote-dir must be a dedicated absolute project path, e.g. /workspace/jacobian-lens"
        )


def ssh_args(config):
    args = ["ssh", "-p", str(config["port"])]
    if config.get("identity"):
        args.extend(["-i", str(Path(config["identity"]).expanduser())])
    return args


def ssh(config, command, *, tty=False):
    return [
        *ssh_args(config),
        *(["-t"] if tty else []),
        f"{config['user']}@{config['host']}",
        "bash -lc " + shlex.quote(command),
    ]


def sync_command(config, direction):
    remote = f"{config['user']}@{config['host']}:{config['remote_dir'].rstrip('/')}"
    command = [
        "rsync",
        "-az",
        "--no-owner",
        "--no-group",
        "--no-perms",
        "--progress",
        "-e",
        shlex.join(ssh_args(config)),
    ]
    if direction == "push":
        for pattern in EXCLUDES:
            command.extend(["--exclude", pattern])
        command.extend([str(ROOT) + "/", remote + "/"])
    else:
        command.extend([remote + "/runs/", str(ROOT / "runs") + "/"])
    return command


def execute(command, dry_run=False):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="action", required=True)
    conf = sub.add_parser("configure", help="save defaults locally (gitignored)")
    conf.add_argument("--host", required=True)
    conf.add_argument("--port", type=int, default=22)
    conf.add_argument("--user", default="root")
    conf.add_argument("--identity", help="local private key path, never copied")
    conf.add_argument("--remote-dir", default="/workspace/jacobian-lens")
    for name in ("push", "pull", "setup", "ssh"):
        sub.add_parser(name)
    status = sub.add_parser("status", help="GPU plus overall/per-method progress")
    status.add_argument("--name", default="tjlens-a40")
    for name in ("run", "smoke", "benchmark", "compose", "logs"):
        command = sub.add_parser(name)
        command.add_argument(
            "--name",
            default={
                "smoke": "tjlens-smoke",
                "benchmark": "tjlens-arithmetic",
                "compose": "gemma4-e4b-composed25",
            }.get(name, "tjlens-a40"),
        )
        if name != "logs":
            command.add_argument(
                "args", nargs=argparse.REMAINDER, help="runner flags after --"
            )
    remote_exec = sub.add_parser("exec")
    remote_exec.add_argument("args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.action == "configure":
        config = {
            key: getattr(args, key)
            for key in ("host", "port", "user", "identity", "remote_dir")
        }
        validate(config)
        if args.dry_run:
            print(json.dumps(config, indent=2))
        else:
            args.config.write_text(json.dumps(config, indent=2) + "\n")
            os.chmod(args.config, 0o600)
            print(f"Saved SSH defaults to {args.config}")
        return
    if not args.config.exists():
        p.error("run configure --host HOST --port PORT first")
    config = json.loads(args.config.read_text())
    validate(config)
    project = shlex.quote(config["remote_dir"])
    prefix = f'export PATH="$HOME/.local/bin:$PATH"; cd {project} && '
    if args.action == "push":
        execute(ssh(config, f"mkdir -p {project}"), args.dry_run)
        execute(sync_command(config, "push"), args.dry_run)
    elif args.action == "pull":
        if not args.dry_run:
            (ROOT / "runs").mkdir(exist_ok=True)
        execute(sync_command(config, "pull"), args.dry_run)
    elif args.action == "setup":
        command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            "if ! command -v uv >/dev/null; then "
            "curl --fail --location --silent --show-error https://astral.sh/uv/install.sh | sh || exit; fi; "
            f"cd {project} && uv sync --locked --extra dev --extra experiment && "
            "uv run --locked pytest -q && nvidia-smi && "
            "uv run --locked python -m experiments.short_hop.check_gpu"
        )
        execute(ssh(config, command), args.dry_run)
    elif args.action == "ssh":
        execute(ssh(config, prefix + "exec bash -i", tty=True), args.dry_run)
    elif args.action == "status":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
            p.error("invalid run name")
        execute(
            ssh(
                config,
                prefix
                + "nvidia-smi && uv run --locked python -m experiments.short_hop.progress --run-dir "
                + shlex.quote(f"runs/{args.name}"),
            ),
            args.dry_run,
        )
    elif args.action == "exec":
        if not args.args:
            p.error("exec needs a command, e.g. exec uv run hf auth login")
        execute(ssh(config, prefix + shlex.join(args.args), tty=True), args.dry_run)
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
            p.error(
                "run name must contain only letters, numbers, underscores or hyphens"
            )
        run_dir = f"runs/{args.name}"
        if args.action == "logs":
            execute(
                ssh(config, prefix + f"tail -n 80 -f {run_dir}/run.log"), args.dry_run
            )
        else:
            extra = args.args[1:] if args.args[:1] == ["--"] else args.args
            command = [
                "uv",
                "run",
                "--locked",
                "--extra",
                "experiment",
                "python",
                "-u",
                "-m",
                "experiments.short_hop.composition"
                if args.action == "compose"
                else "experiments.short_hop.benchmark"
                if args.action == "benchmark"
                else "experiments.short_hop.run",
                "--output-dir",
                run_dir,
            ]
            if args.action == "smoke":
                command.append("--smoke")
            command.extend(extra)
            # flock also protects manual/duplicate launches. PID recorded for inspection;
            # no broad pkill or automatic pod shutdown.
            launch = (
                f"mkdir -p {run_dir} && "
                f"nohup flock -n {run_dir}/process.lock {shlex.join(command)} "
                f">> {run_dir}/run.log 2>&1 < /dev/null &"
            )
            execute(ssh(config, prefix + "( " + launch + " )"), args.dry_run)
            print(
                f"Detached run requested. Check status/logs; results go to {run_dir}/REPORT.md."
            )


if __name__ == "__main__":
    main()
