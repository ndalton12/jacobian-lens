from contextlib import contextmanager

import pytest
import torch

from experiments.short_hop.gemma_checks import check_model_paths
from jlens import ActivationRecorder, from_hf, jacobian_for_prompt
from jlens.fitting import valid_position_mask
from jlens.relp import GEMMA4_RULE_VERSION, _norm_forward, relp_rules, rule_version
from jlens.short_hop import TARGET_TO_SOURCES, default_pairs, smoke_pairs

from .tiny import _ByteTokenizer


def tiny_gemma4(vocab_size=32):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        num_hidden_layers=8,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        global_head_dim=4,
        vocab_size=vocab_size,
        vocab_size_per_layer_input=vocab_size,
        hidden_size_per_layer_input=4,
        num_kv_shared_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 4,
        max_position_embeddings=128,
        sliding_window=4,
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=30.0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    hf = Gemma4ForCausalLM(config).eval()
    with torch.no_grad():
        for layer in hf.model.layers:
            layer.layer_scalar.fill_(0.93)
    return hf


def test_depth_schedule_and_shared_boundary_smoke():
    assert default_pairs(26) == TARGET_TO_SOURCES
    assert default_pairs(42) == {
        13: [12, 11, 9, 5],
        23: [22, 21, 19, 15],
        33: [32, 31, 29, 25],
        41: [40, 39, 37, 33],
    }
    model = from_hf(tiny_gemma4(), _ByteTokenizer())
    assert smoke_pairs(model) == {5: [1, 4]}
    assert rule_version(model) == GEMMA4_RULE_VERSION


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma4_relp_forward_bitexact_gradients_and_restoration(dtype):
    hf = tiny_gemma4().to(dtype)
    model = from_hf(hf, _ByteTokenizer())
    prompt = "abcdefghijk"
    report = check_model_paths(hf, model, prompt)
    assert report["forward_preserved"] and report["restored"]
    assert report["source"] == 1 and report["target"] == 7
    assert report["backward_rule"] == GEMMA4_RULE_VERSION
    kwargs = dict(
        source_layers=[1, 3, 5],
        target_layer=7,
        max_seq_len=10,
        skip_first=1,
        dim_batch=4,
    )
    ordinary, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    q_norm = model.layers[2].self_attn.q_norm.forward
    with relp_rules(model):
        assert model.layers[2].self_attn.q_norm.forward == q_norm
        corrected, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    assert all(torch.isfinite(m).all() for m in corrected.values())
    assert not torch.allclose(ordinary[1], corrected[1])
    again, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    for layer in ordinary:
        torch.testing.assert_close(ordinary[layer], again[layer])
    with pytest.raises(RuntimeError), relp_rules(model):
        raise RuntimeError("interrupted")
    for block in model.layers:
        for module in (
            block.mlp,
            block.act_fn,
            block.per_layer_projection,
            block.post_per_layer_input_norm,
        ):
            assert "forward" not in module.__dict__


def test_gemma4_norm_uses_weight_not_one_plus_weight():
    from types import MethodType

    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    norm = Gemma4RMSNorm(3)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.2, 0.5, 1.3]))
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    expected_forward = norm(x).detach()
    norm.forward = MethodType(_norm_forward(norm.forward, gemma4=True), norm)
    value = norm(x)
    assert torch.equal(value, expected_forward)
    value.sum().backward()
    expected_grad = norm.weight / torch.sqrt(x.detach().square().mean() + norm.eps)
    torch.testing.assert_close(x.grad, expected_grad[None])


def test_ple_product_receives_half_gradient_on_both_factors():
    model = from_hf(tiny_gemma4(), _ByteTokenizer())
    block = model.layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    ple = torch.randn(1, 3, 4, requires_grad=True)
    direction = torch.randn(1, 3, 8)
    with relp_rules(model):
        gate = block.per_layer_input_gate(x)
        activated = block.act_fn(gate)
        product = activated * ple
        projected = block.per_layer_projection(product)
        grad_activated, grad_ple = torch.autograd.grad(
            (projected * direction).sum(), (activated, ple)
        )
    upstream = direction @ block.per_layer_projection.weight
    torch.testing.assert_close(grad_activated, upstream * ple * 0.5)
    torch.testing.assert_close(grad_ple, upstream * activated * 0.5)


@contextmanager
def detach_shared_kv(model):
    # Negative control: detaching the reused state changes derivatives while
    # preserving values. This is precisely the silent bug the oracle must catch.
    def detach(module, args, kwargs):
        states = kwargs["shared_kv_states"]
        for kind, pair in list(states.items()):
            states[kind] = tuple(t.detach() for t in pair)

    handle = model.layers[4].self_attn.register_forward_pre_hook(
        detach, with_kwargs=True
    )
    try:
        yield
    finally:
        handle.remove()


def test_shared_kv_jacobians_match_direct_residual_perturbations():
    model = from_hf(tiny_gemma4(), _ByteTokenizer())
    prompt = "abcdefghijk"
    kwargs = dict(
        source_layers=[1, 3, 5],
        target_layer=7,
        max_seq_len=10,
        skip_first=1,
        dim_batch=4,
    )
    matrices, length, _ = jacobian_for_prompt(model, prompt, **kwargs)
    ids = model.encode(prompt, max_length=10)
    mask = valid_position_mask(length, skip_first=1)
    direction = torch.linspace(-1, 1, model.d_model)
    epsilon = 0.003
    for source in kwargs["source_layers"]:
        outputs = []
        for sign in (-1, 1):

            def perturb(module, inputs, output, sign=sign):
                return output + sign * epsilon * mask[None, :, None] * direction

            hook = model.layers[source].register_forward_hook(perturb)
            try:
                with (
                    torch.no_grad(),
                    ActivationRecorder(model.layers, at=[7]) as recorder,
                ):
                    model.forward(ids)
                    outputs.append(recorder.activations[7][0, mask].float().mean(0))
            finally:
                hook.remove()
        finite_difference = (outputs[1] - outputs[0]) / (2 * epsilon)
        torch.testing.assert_close(
            matrices[source] @ direction, finite_difference, atol=3e-3, rtol=3e-2
        )
    with detach_shared_kv(model):
        wrong, _, _ = jacobian_for_prompt(model, prompt, **kwargs)
    assert not torch.allclose(matrices[1], wrong[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        matrices[5], wrong[5]
    )  # Already-computed KV is fixed for this source.


def test_gemma4_rejects_moe_before_any_patch():
    model = from_hf(tiny_gemma4(), _ByteTokenizer())
    model.layers[-1].enable_moe_block = True
    with pytest.raises(ValueError, match="MoE"), relp_rules(model):
        pass
    assert "forward" not in model.layers[0].input_layernorm.__dict__


def test_gemma4_cache_and_no_cache_logits_agree():
    hf = tiny_gemma4()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        full = hf(ids, use_cache=False).logits[:, -1]
        prefix = hf(ids[:, :-1], use_cache=True)
        cached = hf(
            ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True
        ).logits[:, -1]
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multimodal_gemma4_wrapper_has_same_text_path(dtype):
    from transformers import AutoModelForCausalLM, Gemma4Config

    # The public E4B checkpoint uses the conditional-generation wrapper. The
    # text-only test omits unused vision/audio towers, but keeps its PLE path.
    config = Gemma4Config(text_config=tiny_gemma4().config.to_dict())
    config._attn_implementation = "eager"
    hf = AutoModelForCausalLM.from_config(config).eval().to(dtype)
    model = from_hf(hf, _ByteTokenizer())
    assert model.layout.path == "model.language_model"
    checks = check_model_paths(hf, model, "abcdefghijk")
    assert checks["max_forward_logit_error"] < 1e-5
    assert checks["forward_state_bitexact"]
    assert checks["max_forward_state_error"] == 0
    assert checks["forward_logit_dtype"] == str(dtype)


def test_path_check_matches_projection_shapes_and_accepts_bf16_rounding(monkeypatch):
    hf = tiny_gemma4().to(torch.bfloat16)
    model = from_hf(hf, _ByteTokenizer())
    original_head = hf.lm_head.forward
    monkeypatch.setattr(hf.lm_head, "forward", lambda x: original_head(x) * 100)
    original_unembed = model.unembed

    def rounded_unembed(state):
        # Simulate one BF16 rounding step in the independently computed readout,
        # including the 0.0625-size differences reported on the actual GPU.
        assert state.ndim == 3 and state.shape[1] > 1
        logits = original_unembed(state).clone()
        flat = logits.view(-1)
        index = flat.argmax()
        flat[index] = torch.nextafter(flat[index], flat.new_tensor(float("inf")))
        return logits

    monkeypatch.setattr(model, "unembed", rounded_unembed)
    checks = check_model_paths(hf, model, "abcdefghijk")
    assert checks["max_forward_logit_error"] >= 0.0625
    assert checks["max_forward_state_error"] == 0
    assert checks["forward_logit_atol"] == 2 * torch.finfo(torch.bfloat16).eps


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offset", [1.0, float("nan")])
def test_path_check_still_rejects_wrong_or_nonfinite_readout(
    monkeypatch, dtype, offset
):
    hf = tiny_gemma4().to(dtype)
    model = from_hf(hf, _ByteTokenizer())
    original = model.unembed
    monkeypatch.setattr(model, "unembed", lambda state: original(state) + offset)
    with pytest.raises(RuntimeError, match="differs from full HF logits"):
        check_model_paths(hf, model, "abcdefghijk")


def test_path_check_rejects_internal_path_change_even_with_matching_logits(monkeypatch):
    hf = tiny_gemma4().to(torch.bfloat16)
    model = from_hf(hf, _ByteTokenizer())
    original = hf.forward

    def wrong_full_path(*args, **kwargs):
        # Positive rescaling is almost invisible after the final RMSNorm, but
        # is a genuine mismatch of the residual states we intend to fit.
        hook = model.layers[-1].register_forward_hook(
            lambda module, inputs, output: output * 2, prepend=True
        )
        try:
            return original(*args, **kwargs)
        finally:
            hook.remove()

    monkeypatch.setattr(hf, "forward", wrong_full_path)
    with pytest.raises(RuntimeError, match="differs from full HF block outputs"):
        check_model_paths(hf, model, "abcdefghijk")


def test_same_position_single_probe_matches_direct_block_derivative():
    model = from_hf(tiny_gemma4(), _ByteTokenizer())
    prompt = "abcdefghijk"
    # One valid position removes Hutchinson sampling noise, even though the
    # model still has causal attention, per-layer inputs and shared KV.
    matrices, _, n_valid = jacobian_for_prompt(
        model,
        prompt,
        [1, 5],
        target_layer=7,
        max_seq_len=6,
        skip_first=4,
        dim_batch=4,
        position_reduction="self_hutchinson",
        position_seed=17,
    )
    assert n_valid == 1
    ids = model.encode(prompt, max_length=6)
    for source in (1, 5):
        with ActivationRecorder(
            model.layers, at=[source, 7], start_graph_at=source
        ) as recorder:
            model.forward(ids)
            for row in range(model.d_model):
                gradient = torch.autograd.grad(
                    recorder.activations[7][0, 4, row],
                    recorder.activations[source],
                    retain_graph=True,
                )[0]
                torch.testing.assert_close(
                    matrices[source][row], gradient[0, 4], atol=1e-5, rtol=1e-4
                )
