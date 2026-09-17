# Implementation Brief: J-Lens++ Short-Hop MVP on Gemma 3 1B

## 0. Objective

Implement a minimal, reproducible extension of Anthropic's `jacobian-lens` repository that fits and evaluates **short-hop Jacobian transport maps**

\[
K_{j\leftarrow i}=\mathbb{E}\left[\frac{\partial h_j}{\partial h_i}\right],\qquad i<j,
\]

instead of only maps from an intermediate layer to the final layer.

The MVP should answer one primary research question:

> For controlled tasks with a known hidden intermediate, which layer spans add, preserve, or transform that intermediate representation?

The primary measurement is an **innovation readout** comparing the short-hop transported state against an identity-transport baseline. Reuse Anthropic's fitting code and model adapter as much as possible. Do not build an independent Jacobian framework.

---

## 1. Upstream references

- Anthropic code: <https://github.com/anthropics/jacobian-lens>
- Paper: <https://transformer-circuits.pub/2026/workspace/index.html>
- Model: <https://huggingface.co/google/gemma-3-1b-it>

The upstream repository is a reference implementation and says it is not maintained or accepting contributions. Make these changes in a fork or downstream repository and preserve its Apache-2.0 notices.

---

## 2. Model and runtime assumptions

Use:

```text
model_id = google/gemma-3-1b-it
```

Reasons:

- Small enough for iterative Jacobian experiments.
- Instruction-tuned, so it is more likely than the base checkpoint to solve the controlled evaluation tasks.
- Text-only `Gemma3ForCausalLM` is sufficient; do not load a multimodal wrapper.

Expected architecture values should be read from `model.config`, not hard-coded. For this checkpoint they should be approximately:

```text
num_hidden_layers = 26
hidden_size = 1152
vocab_size = 262144
```

Loading requirements:

1. The user must accept the Gemma license on Hugging Face.
2. Authenticate with `HF_TOKEN` or `huggingface-cli login` / `hf auth login`.
3. Load weights in BF16 on CUDA.
4. Do **not** use 4-bit or 8-bit quantization for the MVP. Quantization introduces avoidable autograd and numerical confounds.
5. Start with `compile=False` and a single GPU. Do not use `device_map="auto"`.

Suggested loading code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jlens

MODEL_ID = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")

model = jlens.from_hf(
    hf_model,
    tokenizer,
    compile=False,
    force_bos=True,
)
```

Verify on startup:

```python
assert model.n_layers == hf_model.config.get_text_config().num_hidden_layers
assert model.d_model == hf_model.config.get_text_config().hidden_size
```

---

## 3. MVP scope

### Implement

1. A reusable short-hop atlas container.
2. Fitting `K_{j<-i}` for a sparse set of layer pairs by reusing `jlens.fit()`.
3. A same-position stochastic Jacobian estimator suitable for local computation analysis.
4. Per-layer activation means for centered transport.
5. Controlled two-hop tasks with explicit intermediate labels.
6. Readouts and quantitative metrics for transported states and innovations.
7. CLI scripts, tests, JSONL/CSV outputs, and a few static plots.

### Explicitly exclude

Do not implement these yet:

- Full all-pairs layer atlas.
- Causal steering or activation interventions.
- Composition/semigroup error experiments.
- Full source-position × target-position Jacobians.
- Learned layer-local decoders.
- Regression transport maps.
- Web UI or modification of Anthropic's D3 visualizer.
- Multi-model support beyond keeping APIs generic where easy.

---

## 4. Layer-pair plan

Gemma 3 1B has 26 residual blocks, indexed `0..25` by the existing hooks. Fit four target layers and four source layers per target:

```python
TARGET_TO_SOURCES = {
    8:  [7, 6, 4, 0],
    14: [13, 12, 10, 6],
    20: [19, 18, 16, 12],
    25: [24, 23, 21, 17],
}
```

This gives hop lengths `{1, 2, 4, 8}` at four depth anchors, for 16 matrices total.

Important optimization: for each fixed target `j`, pass all four source layers to one call of `jlens.fit()`. The upstream implementation computes gradients for multiple source layers in the same backward passes, so do not run one fit per pair.

Approximate storage at `d_model=1152`:

- One FP32 matrix: about 5.1 MiB.
- Sixteen FP32 matrices: about 81 MiB.
- Saved as FP16: about 41 MiB.

---

## 5. Critical estimator change: same-position Hutchinson estimator

### Why the upstream reduction is not ideal here

The upstream estimator injects a cotangent at all valid target positions and then reads the gradient at each source position. In a causal model, this estimates a sum over current and future target positions:

\[
\frac{1}{P}\sum_p\sum_{q\ge p}
\frac{\partial h_{j,q}}{\partial h_{i,p}}.
\]

That is appropriate for eventual verbalizability, but it mixes local state transformation with cross-position broadcasting. For the short-hop MVP, estimate the same-position map:

\[
K^{\mathrm{self}}_{j\leftarrow i}
=
\frac{1}{P}\sum_p
\frac{\partial h_{j,p}}{\partial h_{i,p}}.
\]

Computing every diagonal position block exactly would multiply cost by sequence length. Instead add an unbiased Hutchinson-style estimator that has approximately the same cost as the existing reduction.

### Estimator derivation

For each output dimension `r`, sample independent Rademacher signs `z_p in {-1,+1}` over valid positions. Inject

\[
\bar h_{j,p,r}=z_p.
\]

Autograd returns

\[
g_{p,s}=\sum_q z_q
\frac{\partial h_{j,q,r}}{\partial h_{i,p,s}}.
\]

Then compute

\[
\frac{1}{P}\sum_p z_p g_{p,s}.
\]

Since `E[z_p z_q] = 1[p=q]`, this is an unbiased estimator of the average same-position Jacobian block.

### Required code change

Modify `jlens/fitting.py` while preserving backward compatibility:

```python
PositionReduction = Literal["future_sum", "self_hutchinson"]
```

Add to both `jacobian_for_prompt()` and `fit()`:

```python
position_reduction: PositionReduction = "future_sum"
position_seed: int = 0
```

Keep `future_sum` behavior byte-for-byte equivalent where practical.

For `self_hutchinson`:

1. Generate a deterministic sign tensor of shape `[d_model, n_valid_positions]` on CPU.
2. Seed it from `position_seed` plus a stable SHA-256 hash of the prompt. Do not use Python's randomized `hash()`.
3. For each output-dimension batch, put the corresponding signs into the cotangent instead of ones.
4. After `autograd.grad`, multiply each source-position gradient by the same sign and then average across valid positions.

Pseudocode inside the existing dimension loop:

```python
# Precomputed once per prompt:
# signs: [d_model, n_valid]
row_signs = signs[dim_start : dim_start + n_dims_this_pass].to(target_device)

cotangent.zero_()
cotangent[
    batch_indices[:n_dims_this_pass, None],
    valid_positions[None, :],
    dim_start + batch_indices[:n_dims_this_pass, None],
] = row_signs

# After autograd.grad, for each source-layer grad:
source_signs = row_signs.to(grad.device)
rows = (
    grad[:n_dims_this_pass, positions_on_device, :].float()
    * source_signs[..., None].float()
).mean(dim=1)
```

Checkpoint metadata must include `position_reduction` and `position_seed`; reject incompatible resumes.

The short-hop fitter should default to:

```text
position_reduction = self_hutchinson
```

The original `jlens.fit()` public default must remain:

```text
position_reduction = future_sum
```

---

## 6. Centered transport and activation means

A Jacobian is a local derivative, not a complete state-transition matrix. Do not interpret `K @ h_i` as a direct prediction of `h_j` without centering.

Fit an activation mean for every layer appearing as a source or target:

\[
\mu_l=\mathbb{E}[h_l].
\]

Add:

```text
jlens/stats.py
```

with:

```python
def fit_activation_means(
    model: LensModel,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    max_seq_len: int = 64,
    skip_first: int = 8,
) -> dict[int, torch.Tensor]:
    ...
```

Implementation requirements:

- Use `ActivationRecorder` and one forward pass per prompt.
- Use the same valid-position mask as Jacobian fitting.
- Compute a per-prompt mean over valid positions, then average prompts equally. This matches the upstream estimator's prompt weighting better than weighting all tokens globally.
- Return FP32 CPU vectors of shape `[d_model]`.
- Skip too-short prompts consistently.

For an evaluation activation `h_i`, define:

```python
x_i = h_i.float() - mu_i
h_hat_j = mu_j + K_j_i @ x_i
h_identity_j = mu_j + x_i
innovation_vector = K_j_i @ x_i - x_i
```

Use row-vector PyTorch convention consistently:

```python
transported = x_i @ K_j_i.T
```

---

## 7. Short-hop atlas API

Add:

```text
jlens/short_hop.py
```

Implement a container similar in spirit to `JacobianLens`, but do not overload `JacobianLens` because it assumes a final-layer target and does not record target-layer metadata.

Suggested API:

```python
class ShortHopAtlas:
    jacobians: dict[int, dict[int, torch.Tensor]]
    # jacobians[target_layer][source_layer] = K_{target<-source}

    means: dict[int, torch.Tensor]
    n_prompts_by_target: dict[int, int]
    d_model: int
    n_layers: int
    model_id: str | None
    position_reduction: str
    fit_config: dict[str, Any]

    @property
    def pairs(self) -> list[tuple[int, int]]: ...

    def get(self, source_layer: int, target_layer: int) -> torch.Tensor: ...

    def transport_centered(
        self,
        residual: torch.Tensor,
        source_layer: int,
        target_layer: int,
    ) -> torch.Tensor: ...

    def innovation_vector(
        self,
        residual: torch.Tensor,
        source_layer: int,
        target_layer: int,
    ) -> torch.Tensor: ...

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ShortHopAtlas": ...
```

`transport_centered()` should return the predicted target state, including `mu_j`:

```python
x = residual.float() - means[source]
return means[target] + x @ K.T
```

`innovation_vector()` should return:

```python
x @ K.T - x
```

Validate:

- Every source is `< target`.
- Every matrix is `[d_model, d_model]`.
- All required means exist.
- Layer indices are within `0..n_layers-1`.

Save enough metadata to reproduce the fit, including target/source mapping, model ID, corpus path or corpus hash, seed, maximum sequence length, `skip_first`, `dim_batch`, and estimator mode.

---

## 8. Atlas fitting function

Add:

```python
def fit_short_hop_atlas(
    model: LensModel,
    prompts: Sequence[str],
    *,
    target_to_sources: Mapping[int, Sequence[int]],
    model_id: str | None = None,
    dim_batch: int = 16,
    max_seq_len: int = 64,
    skip_first: int = 8,
    position_reduction: str = "self_hutchinson",
    position_seed: int = 0,
    checkpoint_dir: str | None = None,
) -> ShortHopAtlas:
    ...
```

Implementation:

1. Validate all pairs.
2. Compute activation means once for the union of all source and target layers.
3. For each target layer `j`, call existing `jlens.fit()` once with all sources for that target.
4. Use a separate resumable checkpoint per target, for example:

```text
checkpoints/target_08.pt
checkpoints/target_14.pt
checkpoints/target_20.pt
checkpoints/target_25.pt
```

5. Repackage the returned `JacobianLens.jacobians` into the atlas.
6. Save the atlas matrices as FP16 by default, but perform fitting and evaluation arithmetic in FP32 where practical.

Do not use `JacobianLens.merge()` across different target layers. It has no target metadata and would silently mix semantically different maps.

---

## 9. Fitting corpus

Create:

```text
experiments/short_hop/build_fit_corpus.py
```

Default corpus:

```text
WikiText-103 raw train split
```

Use the `datasets` dependency already present in the upstream development extras. Produce a deterministic JSONL file containing 32 text sequences, each at least 64 Gemma tokens before truncation.

Default fitting settings:

```text
n_prompts = 32
max_seq_len = 64
skip_first = 8
seed = 0
```

The corpus builder should:

1. Stream or iterate nonempty text records.
2. Remove headings and very short fragments where easy.
3. Accumulate text until it reaches at least 64 tokens.
4. Write one JSON object per line:

```json
{"id": "fit-0000", "text": "..."}
```

5. Store a SHA-256 digest of the resulting file in run metadata.

Expose `--n-prompts` so a later run can use 100 prompts without code changes.

---

## 10. Controlled evaluation dataset

Create:

```text
experiments/short_hop/build_eval_cases.py
```

The primary task should be synthetic two-hop lookup with all required facts supplied in the prompt. This avoids relying on world knowledge and gives an exact intermediate label.

Example form:

```text
Facts:
- The dax is associated with red.
- Anything associated with red maps to seven.

Question: What does the dax map to?
Respond with only the final value.
```

Metadata:

```json
{
  "id": "lookup-0001",
  "task_type": "two_hop_lookup",
  "user_prompt": "Facts: ...",
  "intermediate": "red",
  "answer": "seven"
}
```

Requirements:

- Generate at least 64 deterministic cases.
- Use nonce entities for the first hop.
- Randomize fact order and include irrelevant distractor facts in at least half the cases.
- Ensure intermediate and answer labels are distinct.
- Prefer labels that tokenize as exactly one token in answer context.
- At build time, validate tokenization with Gemma's tokenizer and record:

```json
{
  "intermediate_token_id": 123,
  "answer_token_id": 456
}
```

- If a candidate label is not one token, replace it from a candidate pool rather than implementing multi-token scoring in the MVP.

Before lens analysis, run greedy generation and mark whether the model solved the case. Aggregate headline metrics on the solved subset, while retaining all cases in outputs.

For the instruction-tuned model, create input IDs using the tokenizer's chat template:

```python
messages = [{"role": "user", "content": case["user_prompt"]}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

Do not convert the chat template to text and then tokenize again; that can duplicate special tokens. The evaluation path should accept token IDs directly instead of relying on `HFLensModel.encode()`.

Probe the final input position (`position=-1`), which is the position predicting the first answer token.

---

## 11. Readouts

For each solved case and each pair `(i, j)`, record `h_i` and `h_j` at the probe position in one model forward pass.

Compute:

### A. Actual local readout

```python
actual_local_logits_j = model.unembed(h_j)
```

This asks what is directly vocabulary-aligned in the actual layer-`j` state under the model's final norm and LM head.

### B. Centered transported-state readout

```python
h_hat_j = atlas.transport_centered(h_i, i, j)
transported_logits = model.unembed(h_hat_j)
```

This asks what the average short-hop Jacobian predicts should become locally readable at `j`.

### C. Identity-transport baseline

```python
x_i = h_i.float() - mu_i
h_identity_j = mu_j + x_i
identity_logits = model.unembed(h_identity_j)
```

This controls for merely carrying the centered residual forward without a learned short-hop transformation.

### D. Primary innovation score

Use a **difference of complete readouts**:

```python
innovation_logits = transported_logits - identity_logits
```

This is the primary MVP score. It measures which vocabulary logits are added or suppressed by replacing identity transport with `K`.

Do not call `model.unembed(innovation_vector)` as the primary score. `model.unembed()` contains RMSNorm, and normalizing a standalone difference vector is not equivalent to the logit effect of that difference.

### E. Optional linear innovation diagnostic

For comparison only, expose the LM head without the final norm:

```python
linear_innovation_logits = lm_head(innovation_vector)
```

If implementing this, add a public `linear_unembed()` method to `HFLensModel` rather than reaching into private fields from experiment code. Do not apply final logit softcapping to a logit-difference vector.

### F. Actual update comparison

```python
actual_update = h_j.float() - h_i.float()
predicted_update = atlas.innovation_vector(h_i, i, j)
```

Record cosine similarity and relative norm. This is only a diagnostic because the actual update also contains affine and context-dependent effects not captured by the average Jacobian.

---

## 12. Metrics

For every case/pair/readout, store:

- Intermediate token logit.
- Answer token logit.
- Intermediate token rank.
- Answer token rank.
- `intermediate_logit - answer_logit`.
- Top 20 decoded tokens and scores.
- Source layer, target layer, and hop length.
- Whether the base model solved the case.

Efficient rank computation without sorting the full vocabulary:

```python
def token_rank(logits: torch.Tensor, token_id: int) -> int:
    score = logits[token_id]
    return int((logits > score).sum().item()) + 1
```

Also record transport quality:

```text
relative_state_error = ||h_hat_j - h_j|| / (||h_j - mu_j|| + eps)
update_cosine = cosine(predicted_update, actual_update)
transported_vs_actual_logit_cosine
transported_vs_actual_top20_jaccard
```

Primary aggregate metrics:

1. Median intermediate rank under `innovation_logits` for each `(i, j)`.
2. Mean innovation margin:

\[
\text{innovation}[intermediate]-\text{innovation}[answer].
\]

3. Fraction of solved cases where the intermediate ranks above the final answer in innovation logits.
4. Comparison against the identity baseline and actual local readout.
5. Best target depth and hop length for intermediate emergence.

Do not report only cherry-picked token lists. The implementation is incomplete without aggregate metrics across the generated dataset.

---

## 13. Output layout

Use a run directory such as:

```text
runs/gemma3_1b_short_hop_mvp/
├── config.json
├── fit_prompts.jsonl
├── eval_cases.jsonl
├── checkpoints/
│   ├── target_08.pt
│   ├── target_14.pt
│   ├── target_20.pt
│   └── target_25.pt
├── short_hop_atlas.pt
├── generation_results.jsonl
├── pair_results.jsonl
├── aggregate_metrics.csv
└── plots/
    ├── intermediate_rank_heatmap.png
    ├── innovation_margin_heatmap.png
    ├── rank_by_target_layer.png
    └── transport_error_by_hop.png
```

`config.json` must contain all parameters and software versions, including:

```text
model_id
model revision if available
transformers version
torch version
CUDA device name
dtype
target_to_sources
position_reduction
position_seed
fit corpus SHA-256
n_prompts
max_seq_len
skip_first
dim_batch
eval dataset seed
```

---

## 14. CLI scripts

Create these scripts under `experiments/short_hop/`:

### Build fitting corpus

```bash
python -m experiments.short_hop.build_fit_corpus \
  --model google/gemma-3-1b-it \
  --output runs/gemma3_1b_short_hop_mvp/fit_prompts.jsonl \
  --n-prompts 32 \
  --seq-len 64 \
  --seed 0
```

### Build evaluation cases

```bash
python -m experiments.short_hop.build_eval_cases \
  --model google/gemma-3-1b-it \
  --output runs/gemma3_1b_short_hop_mvp/eval_cases.jsonl \
  --n-cases 64 \
  --seed 0
```

### Fit atlas

```bash
python -m experiments.short_hop.fit_atlas \
  --model google/gemma-3-1b-it \
  --prompts runs/gemma3_1b_short_hop_mvp/fit_prompts.jsonl \
  --output runs/gemma3_1b_short_hop_mvp/short_hop_atlas.pt \
  --checkpoint-dir runs/gemma3_1b_short_hop_mvp/checkpoints \
  --max-seq-len 64 \
  --skip-first 8 \
  --dim-batch 16 \
  --position-reduction self_hutchinson \
  --seed 0
```

### Evaluate

```bash
python -m experiments.short_hop.evaluate \
  --model google/gemma-3-1b-it \
  --atlas runs/gemma3_1b_short_hop_mvp/short_hop_atlas.pt \
  --cases runs/gemma3_1b_short_hop_mvp/eval_cases.jsonl \
  --output-dir runs/gemma3_1b_short_hop_mvp
```

### Plot

```bash
python -m experiments.short_hop.plot_results \
  --results runs/gemma3_1b_short_hop_mvp/pair_results.jsonl \
  --output-dir runs/gemma3_1b_short_hop_mvp/plots
```

All scripts must support `--help`, deterministic seeds, and useful logging.

---

## 15. Performance guidance

The upstream estimator costs one forward pass and approximately

```text
ceil(d_model / dim_batch)
```

backward passes per prompt and target layer.

For Gemma 3 1B with `d_model=1152`:

```text
dim_batch=16 -> 72 backward passes per prompt/target
dim_batch=32 -> 36 backward passes per prompt/target
```

Start with:

```text
max_seq_len=64
n_prompts=4
one target layer
one or two source layers
```

for a smoke test. Then run the full four-target, 32-prompt MVP.

If memory permits, `dim_batch=32` will reduce Python/autograd-loop overhead, but it replicates the prompt 32 times and may OOM. The implementation should fail with a clear message suggesting a smaller `--dim-batch`; automatic OOM tuning is not required.

Evaluation should batch transported states across all pairs for a case before unembedding, and should not retain full vocabulary logits in output files. Store metrics and top-k values only.

---

## 16. Tests

Preserve all existing upstream tests.

Add:

```text
tests/test_short_hop.py
tests/test_self_hutchinson.py
tests/test_stats.py
```

### Required unit tests

1. **Atlas validation and round trip**
   - Save/load preserves maps, means, metadata, and pair ordering.
   - Invalid source/target indices raise clear errors.

2. **Exact short-hop map on `TinyDecoder`**
   - For the existing linear residual model, verify `K_{j<-i}` equals the product of intervening block Jacobians.
   - Verify row/column orientation by comparing `residual @ K.T` with the actual linear forward map.

3. **Innovation identity case**
   - For `K=I`, `innovation_vector()` and innovation logits are zero up to numerical tolerance.

4. **Activation means**
   - Correct shapes, FP32 CPU dtype, deterministic values, and prompt weighting.

5. **Self-Hutchinson estimator**
   - Build a tiny causal token-mixing model where cross-position derivatives are nonzero.
   - Compute the exact average diagonal position Jacobian by explicitly looping over positions.
   - Average the Hutchinson estimator over a deterministic set of sign probes or enough fixed seeds to match the exact value within a stated tolerance.
   - Verify `future_sum` and `self_hutchinson` differ on this model.

6. **Backward compatibility**
   - Existing `fit()` calls without the new parameter behave as `future_sum`.
   - Existing checkpoints still load where metadata is absent; treat missing reduction as `future_sum`.

### Optional gated integration test

Add a test skipped unless both conditions are met:

```text
RUN_GEMMA_TESTS=1
HF_TOKEN is available
```

It should load Gemma, fit one prompt for one pair at short sequence length, and check only shapes, finiteness, serialization, and evaluation execution. Do not run this in normal CI.

---

## 17. Acceptance criteria

The MVP is complete when all of the following are true:

- [ ] Existing Anthropic tests pass unchanged.
- [ ] New unit tests pass on CPU using tiny models.
- [ ] Gemma 3 1B loads through the existing `jlens.from_hf()` adapter without a custom layout.
- [ ] A smoke fit produces a finite `K_{j<-i}` and resumes from a checkpoint.
- [ ] The full configured run produces 16 short-hop maps and activation means.
- [ ] Evaluation uses chat-template token IDs without double tokenization.
- [ ] Results include actual, transported, identity, and innovation readouts.
- [ ] Aggregate intermediate-rank metrics are generated over the solved subset.
- [ ] Output metadata is sufficient to reproduce the run.
- [ ] No claim of “intermediate computation” is based solely on top-k anecdotes.

---

## 18. Expected first analysis

The first report should be deliberately modest. It should answer:

1. Does centered short-hop transport predict actual target-layer states better than identity transport?
2. Does the innovation readout elevate the labelled intermediate above the final answer at any depth?
3. Are effects consistent across cases, or driven by a few examples?
4. Are hop-1 maps dominated by identity while hop-4 or hop-8 maps show stronger semantic change?
5. Does the same-position estimator produce clearer results than the upstream future-sum estimator on a small comparison run?

A positive MVP result would look like:

- Intermediate-token innovation rank improves consistently in middle-layer targets.
- The improvement is stronger than identity and source logit-lens baselines.
- Transported states have nontrivial alignment with actual target states.
- Results survive aggregation across the synthetic dataset.

A negative result is also useful if:

- Transport error is high.
- Innovation ranks are unstable across corpus seeds.
- Raw unembedding at intermediate target layers is geometrically misaligned.

Record these failure modes rather than hiding them.

---

## 19. Implementation order

Implement in this order:

1. Add `position_reduction` with backward-compatible `future_sum` behavior.
2. Add and test `self_hutchinson` on tiny causal models.
3. Add activation-mean fitting.
4. Add `ShortHopAtlas` and serialization.
5. Add sparse-target fitting CLI and smoke run.
6. Add controlled dataset generation and model-solved filtering.
7. Add batched evaluation/readouts and JSONL outputs.
8. Add aggregate metrics and plots.
9. Run the four-target Gemma MVP.

Do not start by fitting the full Gemma atlas before CPU tests and the one-target smoke run pass.
