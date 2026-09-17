"""Check the actual PyTorch CUDA runtime and BF16 forward/backward execution."""

import argparse
import json
import subprocess

import torch


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable to torch {torch.__version__} (wheel runtime {torch.version.cuda}). "
            "Check GPU access and host driver compatibility; a container toolkit alone is insufficient."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment needs BF16 support, e.g. an A40.")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    # Exercise cuBLAS, normalization, and autograd, rather than trusting a version label.
    with torch.enable_grad():
        x = torch.ones(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = x @ torch.eye(32, device="cuda", dtype=torch.bfloat16)
        y = y * torch.rsqrt(y.float().square().mean(-1, keepdim=True) + 1e-6)
        y.sum().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
            raise RuntimeError(
                "CUDA BF16 forward/backward check produced nonfinite values"
            )
    result = dict(
        torch=str(torch.__version__),
        torch_cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        driver=driver,
        bf16_forward_backward="passed",
    )
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(check_gpu(), indent=2))


if __name__ == "__main__":
    main()
