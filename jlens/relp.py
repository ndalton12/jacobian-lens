"""Dense Gemma 3/4 R-Lens rules, following Blank, Bhatia & Nanda (2026).

https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers

Residual RMSNorm denominator detached; GELU/SiLU nonlinear factor detached;
gated product relevance split equally. Attention and q/k norms are untouched.
This is a Gemma implementation of the published rules, not an authors' artifact.
"""

import math
from contextlib import contextmanager
from types import MethodType

import torch

RULE_VERSION = "gemma3-relp-ln-identity-half-v1"
GEMMA4_RULE_VERSION = "gemma4-dense-relp-ln-identity-half-ple-v1"


def rule_version(model):
    """Validate architecture before fitting; never silently apply Gemma 3 rules."""
    names = {type(block).__name__ for block in model.layers}
    if names == {"Gemma3DecoderLayer"}:
        return RULE_VERSION
    if names == {"Gemma4TextDecoderLayer"}:
        if any(block.enable_moe_block for block in model.layers):
            raise ValueError("Gemma 4 MoE R-Lens is not supported; use dense E4B/E2B")
        return GEMMA4_RULE_VERSION
    raise ValueError(
        "R-Lens supports dense Gemma3DecoderLayer/Gemma4TextDecoderLayer only"
    )


class _FixedFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, value, factor):
        ctx.save_for_backward(factor)
        ctx.dtype = x.dtype
        return value

    @staticmethod
    def backward(ctx, grad):
        (factor,) = ctx.saved_tensors
        return (grad.float() * factor.float()).to(ctx.dtype), None, None


class _HalfProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad * b * 0.5, grad * a * 0.5


def _norm_forward(original, *, gemma4=False):
    def forward(module, x):
        with torch.no_grad():
            value = original(x)
            factor = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + module.eps)
            if gemma4:
                # Gemma 4 multiplies by weight, not 1 + weight (Gemma 3).
                if module.with_scale:
                    factor = factor * module.weight.float()
            else:
                factor = factor * (1 + module.weight.float())
        return _FixedFactor.apply(x, value, factor)

    return forward


class _HalfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * 0.5


def _ple_projection_forward(original):
    def forward(module, x):
        # x = activation(gate(h)) * per_layer_input. Halving the gradient of
        # this product halves each factor's contribution, exactly as HalfProduct.
        return original(_HalfGradient.apply(x))

    return forward


def _activation_forward(original, kind):
    def forward(module, x):
        with torch.no_grad():
            value = original(x)
            factor = _activation_factor(x, kind)
        return _FixedFactor.apply(x, value, factor)

    return forward


def _activation_factor(x, kind):
    x = x.float()
    if kind == "silu":
        return x.sigmoid()
    if kind == "gelu":
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    if kind in ("gelu_pytorch_tanh", "gelu_new", "gelu_fast"):
        return 0.5 * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3)))
        )
    raise ValueError(f"unsupported R-Lens activation {kind!r}")


def _mlp_forward(kind):
    def forward(module, x):
        gate, up = module.gate_proj(x), module.up_proj(x)
        with torch.no_grad():
            value = module.act_fn(gate)
            factor = _activation_factor(gate, kind)
        activated = _FixedFactor.apply(gate, value, factor)
        return module.down_proj(_HalfProduct.apply(activated, up))

    return forward


@contextmanager
def relp_rules(model):
    """Temporarily install dense Gemma 3/4 rules; restore even after errors.

    Only explicit residual block norm attributes are patched. A forward invariant
    check on real input should accompany fitting (the experiment runner does this).
    """
    version = rule_version(model)
    gemma4 = version == GEMMA4_RULE_VERSION
    patched = []

    def replace(module, fn):
        # Restore instance attribute presence too, not only callable behavior.
        patched.append((module, module.__dict__.get("forward")))
        module.forward = MethodType(fn, module)

    try:
        for block in model.layers:
            norm_names = [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ]
            if gemma4 and block.hidden_size_per_layer_input:
                norm_names.append("post_per_layer_input_norm")
            for name in norm_names:
                norm = getattr(block, name)
                if type(norm).__name__ != (
                    "Gemma4RMSNorm" if gemma4 else "Gemma3RMSNorm"
                ):
                    raise ValueError(
                        f"unsupported residual norm: {type(norm).__name__}"
                    )
                replace(norm, _norm_forward(norm.forward, gemma4=gemma4))
            mlp = block.mlp
            kind = mlp.config.hidden_activation
            _activation_factor(torch.zeros(1), kind)  # fail before fitting
            replace(mlp, _mlp_forward(kind))
            if gemma4 and block.hidden_size_per_layer_input:
                replace(block.act_fn, _activation_forward(block.act_fn.forward, kind))
                replace(
                    block.per_layer_projection,
                    _ple_projection_forward(block.per_layer_projection.forward),
                )
        yield
    finally:
        for module, original in reversed(patched):
            if original is None:
                del module.forward
            else:
                module.forward = original
