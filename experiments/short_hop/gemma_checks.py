"""Architecture provenance and lightweight real-checkpoint forward/gradient checks."""

import hashlib

import torch

from jlens import ActivationRecorder
from jlens.relp import relp_rules, rule_version

RESIDUAL_SEMANTICS = (
    "Derivative with respect to a block output: token-derived per-layer inputs and "
    "KV computed before that output are held fixed. KV computed downstream stays "
    "in the autograd graph. The residual alone is not the model's full state."
)


def architecture_metadata(model):
    config = model.layers[0].config
    return dict(
        decoder_class=type(model.layers[0]).__name__,
        num_hidden_layers=model.n_layers,
        hidden_size=model.d_model,
        num_kv_shared_layers=getattr(config, "num_kv_shared_layers", 0),
        hidden_size_per_layer_input=getattr(config, "hidden_size_per_layer_input", 0),
        residual_jacobian_semantics=RESIDUAL_SEMANTICS,
    )


def check_model_paths(hf, model, prompt, *, max_seq_len=16):
    """Check wrapper logits, exact R forward preservation, and a finite VJP.

    The VJP starts before the shared-KV providers when available, checking an
    actual-checkpoint backward pass across the sharing boundary before fitting.
    """
    rule = rule_version(model)
    ids = model.encode(prompt, max_length=min(max_seq_len, 16))
    final = model.n_layers - 1
    providers = [
        i
        for i, layer in enumerate(model.layers)
        if getattr(layer.self_attn, "store_full_length_kv", False)
    ]
    source = max(0, min(providers) - 1) if providers else max(0, final - 2)
    with torch.no_grad():
        with ActivationRecorder(model.layers, at=[final]) as recorder:
            model.forward(ids)
            reference_state = recorder.activations[final].detach().clone()
        # Keep the same [batch, sequence, hidden] projection shape as HF. A
        # single-token GEMV and a sequence GEMM can select different CUDA kernels
        # and round differently in BF16 even with identical residual states.
        bare_logits_native = model.unembed(reference_state)
        with ActivationRecorder(model.layers, at=[final]) as recorder:
            # This is an unpadded prompt; match the bare path's absent mask.
            full_logits_native = hf(input_ids=ids, use_cache=False).logits
            full_state = recorder.activations[final]
        state_error = (reference_state.float() - full_state.float()).abs().max().item()
        # Check the actual decoder path independently of the output projection.
        # Do not allow a BF16 logit tolerance to hide a changed internal path.
        if not torch.equal(reference_state, full_state):
            raise RuntimeError(
                "bare text path differs from full HF block outputs "
                f"(max error {state_error}); refusing fit"
            )
        bare_logits = bare_logits_native.float()
        full_logits = full_logits_native.float()
        # Casting BF16 results to float does not restore their lost precision.
        # Allow two rounding units at unit scale (and proportionally above it),
        # retaining the previous tighter threshold for float32 computations.
        tolerance = max(
            1e-3,
            2 * torch.finfo(bare_logits_native.dtype).eps,
            2 * torch.finfo(full_logits_native.dtype).eps,
        )
        max_error = (bare_logits - full_logits).abs().max().item()
        if not torch.allclose(bare_logits, full_logits, atol=tolerance, rtol=tolerance):
            raise RuntimeError(
                "bare text path differs from full HF logits "
                f"(max error {max_error}, dtype {bare_logits_native.dtype}, "
                f"atol/rtol {tolerance}); refusing fit"
            )
    norms = {}

    def vjp(name):
        with (
            torch.enable_grad(),
            ActivationRecorder(
                model.layers, at=[source, final], start_graph_at=source
            ) as recorder,
        ):
            model.forward(ids)
            target = recorder.activations[final]
            if not torch.equal(reference_state, target):
                raise RuntimeError(f"{name} changed block forward outputs")
            gradient = torch.autograd.grad(
                target[:, -1, 0].sum(), recorder.activations[source]
            )[0]
            if not torch.isfinite(gradient).all() or not gradient.float().norm() > 0:
                raise RuntimeError(f"{name}: nonfinite/zero residual gradient")
            norms[name] = gradient.float().norm().item()

    vjp("autograd")
    with relp_rules(model):
        vjp("R")
    with torch.no_grad(), ActivationRecorder(model.layers, at=[final]) as recorder:
        model.forward(ids)
        if not torch.equal(reference_state, recorder.activations[final]):
            raise RuntimeError("R rules were not restored")
    return dict(
        backward_rule=rule,
        source=source,
        target=final,
        n_tokens=ids.shape[1],
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        max_forward_logit_error=max_error,
        max_forward_state_error=state_error,
        forward_state_bitexact=True,
        forward_logit_dtype=str(bare_logits_native.dtype),
        forward_logit_atol=tolerance,
        forward_logit_rtol=tolerance,
        gradient_norms=norms,
        forward_preserved=True,
        restored=True,
    )
