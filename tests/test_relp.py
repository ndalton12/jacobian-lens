import pytest
import torch

from jlens import from_hf, jacobian_for_prompt
from jlens.relp import _activation_factor, _HalfProduct, relp_rules

from .tiny import _ByteTokenizer


def tiny_gemma(vocab_size=32):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(
        num_hidden_layers=3,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=vocab_size,
        max_position_embeddings=128,
        sliding_window=16,
        layer_types=["full_attention"] * 3,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(3)
    return Gemma3ForCausalLM(config).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma_relp_preserves_forward_changes_backward_restores(dtype):
    hf = tiny_gemma().to(dtype)
    model = from_hf(hf, _ByteTokenizer())
    prompt = "abcdefghi"
    ids = model.encode(prompt)
    original = model.forward(ids).last_hidden_state.detach().clone()
    kwargs = dict(source_layers=[0], max_seq_len=10, skip_first=2, dim_batch=4)
    j, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    q_norm = model.layers[1].self_attn.q_norm.forward
    with relp_rules(model):
        assert torch.equal(model.forward(ids).last_hidden_state, original)
        assert model.layers[1].self_attn.q_norm.forward == q_norm
        r, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    assert torch.isfinite(r[0]).all()
    assert not torch.allclose(j[0], r[0], atol=1e-4, rtol=1e-4)
    restored, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    torch.testing.assert_close(j[0], restored[0])
    assert "forward" not in model.layers[1].mlp.__dict__
    with pytest.raises(RuntimeError), relp_rules(model):
        raise RuntimeError("deliberate interruption")
    assert "forward" not in model.layers[1].mlp.__dict__


def test_gelu_factor_zero_is_finite():
    assert _activation_factor(torch.tensor([0.0]), "gelu_pytorch_tanh").item() == 0.5


def test_relp_norm_and_half_rule_gradients():
    model = from_hf(tiny_gemma(), _ByteTokenizer())
    norm = model.layers[1].input_layernorm
    x = torch.randn(2, 8, requires_grad=True)
    with relp_rules(model):
        gradient = torch.autograd.grad(norm(x).sum(), x)[0]
    expected = torch.rsqrt(x.detach().square().mean(-1, keepdim=True) + norm.eps) * (
        1 + norm.weight
    )
    torch.testing.assert_close(gradient, expected.expand_as(x))
    a, b = torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)
    ga, gb = torch.autograd.grad(_HalfProduct.apply(a, b).sum(), (a, b))
    torch.testing.assert_close(ga, 0.5 * b.detach())
    torch.testing.assert_close(gb, 0.5 * a.detach())
