"""Shared configuration, provenance, and I/O for the experiment."""

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

import jlens

MODEL_ID = "google/gemma-3-1b-it"


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--seed", type=int, default=0)
    return result


def model_args(result):
    result.add_argument("--model", default=MODEL_ID)
    result.add_argument("--revision", default="main")


def fit_args(result):
    result.add_argument("--max-seq-len", type=int, default=64)
    result.add_argument("--skip-first", type=int, default=8)
    result.add_argument("--dim-batch", type=int, default=16)
    result.add_argument(
        "--position-reduction",
        choices=["self_hutchinson", "future_sum"],
        default="self_hutchinson",
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # TF32 is disabled for reproducibility of FP32 transport/readout diagnostics.
    torch.backends.cuda.matmul.allow_tf32 = False
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_baseline(lens, path, metadata):
    from jlens.fitting import _atomic_save

    matrices = {
        layer: value.to(torch.float16) for layer, value in lens.jacobians.items()
    }
    if not all(torch.isfinite(value).all() for value in matrices.values()):
        raise FloatingPointError("baseline overflows FP16 storage")
    _atomic_save(
        dict(
            J=matrices,
            n_prompts=lens.n_prompts,
            d_model=lens.d_model,
            source_layers=lens.source_layers,
            metadata=metadata,
        ),
        str(path),
    )


def load_baseline(path, *, expected_metadata):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "metadata" not in state:
        raise ValueError(
            f"{path}: missing experiment provenance; use artifacts from the pilot runner"
        )
    for key, expected in expected_metadata.items():
        if state["metadata"].get(key) != expected:
            raise ValueError(f"{path}: incompatible baseline metadata for {key}")
    return jlens.JacobianLens(
        state["J"], n_prompts=state["n_prompts"], d_model=state["d_model"]
    )


def load_tokenizer(model=MODEL_ID, revision="main"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


def load_model(model=MODEL_ID, revision="main"):
    from transformers import AutoModelForCausalLM

    from experiments.short_hop.check_gpu import check_gpu

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment needs a CUDA GPU; CPU tests work without one."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a GPU supporting BF16 (e.g. A40).")
    logging.getLogger(__name__).info("GPU preflight: %s", check_gpu())
    tokenizer = load_tokenizer(model, revision)
    hf = (
        AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    wrapped = jlens.from_hf(hf, tokenizer, compile=False, force_bos=True)
    config = hf.config.get_text_config()
    assert wrapped.n_layers == config.num_hidden_layers
    assert wrapped.d_model == config.hidden_size
    return hf, wrapped


def provenance(hf=None):
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    source_files = [root / "pyproject.toml", root / "uv.lock"]
    for directory in ("jlens", "experiments", "scripts"):
        source_files.extend(sorted((root / directory).rglob("*.py")))
    for path in source_files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return dict(
        source_sha256=digest.hexdigest(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        transformers=version("transformers"),
        datasets=version("datasets"),
        cuda=torch.version.cuda,
        cuda_device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        dtype="bfloat16",
        attention="eager",
        compile=False,
        model_revision=getattr(hf.config, "_commit_hash", None) if hf else None,
        git_commit=commit.stdout.strip() or None,
        git_dirty=bool(dirty.stdout.strip()),
    )


def chat_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return ids if isinstance(ids, torch.Tensor) else ids["input_ids"]


def answer_token(tokenizer, prompt, label):
    """Validate a one-token continuation at the actual assistant boundary.

    Rendering text here is solely a tokenization validation. Model evaluation
    uses chat_ids directly and never re-tokenizes the rendered chat prompt.
    """
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    before = chat_ids(tokenizer, prompt)[0].tolist()
    after = tokenizer(prefix + label, add_special_tokens=False)["input_ids"]
    if after[: len(before)] != before or len(after) != len(before) + 1:
        raise ValueError(f"{label!r} is not a single answer-context token")
    return int(after[-1])
