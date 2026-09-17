# jlens — Jacobian lens

> **Experimental fork.** The upstream project is a reference implementation marked
> as unmaintained and not accepting contributions. This fork adds the TJ/TR-Lens
> experiments described [below](#fork-experiments-tj-lens-and-tr-lens).

Upstream companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and
position into the final-layer basis, then decodes it with the model's own
unembedding into a ranked list of vocabulary tokens.

The transport is the average input–output Jacobian over a text corpus:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[∂h_final / ∂h_l]
```

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the precise estimator
(cotangents summed over target positions, then averaged over source positions)
is documented in the [`jlens.fitting`](jlens/fitting.py) module docstring.

This repo fits the lens on open-weights decoder transformers, applies it, and
renders the interactive layer × position view shown below. Examples use Qwen;
other HuggingFace decoders adapt cleanly.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

*The ASCII-face example: selecting the `^` (nose) position shows the lens
reading out "nose" at mid layers, although the word never appears in the
prompt.*

## Install

```bash
uv sync --locked --extra dev --extra experiment
```

## Usage

### Apply

To apply a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model, load (or fit) a lens, apply it at a few layers, and render a slice page
like the one above.

Reading a slice page:

- Each cell shows the lens top-1 word at that (position, layer); the
  superscript is its rank over the full vocabulary.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

## License and data

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
## Fork experiments: TJ-Lens and TR-Lens

**Status: paused after exploratory pilots.** We tested whether short-hop transport
could reveal reasoning intermediates more clearly than J-Lens, R-Lens, or logit
lens. The experiments found some rank improvements, but no convincing practical
advantage over R-Lens or the readouts of the model's actual later states. These
are results from this fork, not results or claims of the upstream paper.

The [experiment guide](experiments/short_hop/README.md) contains commands, exact
estimators, architecture conventions, and resume instructions. Everything uses
`uv`; the original J-Lens API defaults are preserved.

### What we tested

J-Lens uses an averaged derivative map from a source layer to the final layer.
R-Lens uses modified backward rules to construct its readout maps. Our
**transport J-Lens (TJ)** instead fits a short-hop map `K` between intermediate
layers; **TR** fits the corresponding short-hop map with R-style backward rules.

We ran two distinct versions. In column-vector notation:

1. **Original short-hop MVP:** predict a later state,
   `h_hat_j = mean_j + K @ (h_i - mean_i)`, then apply the **logit lens** directly:
   `unembed(h_hat_j)`. This does **not** include a J/R map from `j` to the final
   layer. We also tested a *change-only* score: subtract the word scores of an
   identity-transport baseline, `unembed(mean_j + h_i - mean_i)`.
2. **Composed-map follow-up:** apply the target layer's J/R map before decoding:
   `TJ = unembed(J_j @ K @ h_i)` and `TR = unembed(R_j @ K_R @ h_i)`.
   Centered variants substitute `h_hat_j` for `K @ h_i`. There is no normalization
   between maps and **no change-only subtraction** in this follow-up.

Here `unembed` means the model's final normalization and vocabulary output head
(including any logit softcap), not the remaining transformer layers.

The composition hypothesis is that averaging and multiplication do not commute:
`E[B @ A] != E[B] @ E[A]` in general. Separately averaged stages could therefore
give a different, potentially more useful readout than a direct averaged map.
The original MVP did not test that hypothesis; the composed follow-up did, with
the estimator qualifications below.

### Experiments and results

All rank comparisons concern the labelled **intermediate**, not answer accuracy.
For `(2 + 3) × 4`, that is **5**, not **20**. Rank 1 is best. An improvement
factor above 1 means a better rank on geometric average; it is separate from the
fraction of individual questions won.

| Experiment | Setup | Outcome |
| --- | --- | --- |
| Lookup pilot | Gemma 3 1B; controlled two/three-hop questions; J/R/TJ/TR | Only 23/128 answers correct, including 9 solved test cases. Inconclusive; change-only TJ/TR performed poorly. |
| Arithmetic competence checks | Separate 32-question practice set; 80% gate | Gemma 3 1B scored 2/32 and Gemma 3 4B scored 8/32; full arithmetic comparisons were not run for those checks. Gemma 4 E4B passed with 29/32. |
| E4B full arithmetic pilot | 32 fitting prompts per method; 192 arithmetic questions | 187/192 correct; 125/128 test questions correct. Change-only TJ/TR were substantially worse than J/R. TJ predicted-state improved over J by 1.81× (95% interval 1.07–3.24×), but did not establish an advantage over R (0.77×; 0.49–1.27×). |
| Fixed-source short-hop comparison | Source 25, target 33; same 125 solved test questions; no refitting | Median intermediate ranks: logit lens 113,362; R at 25 920; TJ predicted-state 4,890; TR predicted-state 5,831. None reached top 5 on any case. |
| Composed-map comparison | 25 → 33 → final; J/R at both actual layers 25 and 33 | Composition improved TJ over direct J at 25 by 1.59×, winning 107/125 cases. Composed TR was not clearly better than direct R at 25: 0.90× (0.71–1.13×), despite winning 86/125 cases. Centering hurt both compositions. |

The final experiment's key diagnostic was **actual versus predicted layer 33**:

| Readout | Median intermediate rank ↓ | Top-5 cases |
| --- | ---: | ---: |
| J at actual layer 25 | 7,389 | 0/125 |
| R at actual layer 25 | 920 | 0/125 |
| Composed TJ, matrix product | 3,173 | 0/125 |
| Composed TR, matrix product | 1,340 | 0/125 |
| J at actual layer 33 | 11 | 52/125 |
| R at actual layer 33 | 4 | 76/125 |

Actual layer 33 benefits from eight more layers of real model computation; it is
a diagnostic baseline, not an equal-compute alternative. Transport did not
recover its clear intermediate signal from layer 25. The modest improvement over
direct J is a sign that composition changes the readout usefully, but the
intermediate remained buried and poorly distinguished from control labels.

### Evaluation conventions and limitations

- The arithmetic set pairs problems with the same answer but different
  intermediates, each in symbolic and verbal wording. Families stay together
  across selection/test splits and bootstrap resampling. Whole-token number/word
  aliases are fixed before scoring and handled identically across methods.
- Main arithmetic questions use the model's chat template, request only the
  answer, and do not enable thinking mode. Lenses inspect the **final input
  token**, before any answer tokens, at zero-based block-output layer indices.
- The full pilot selects layers on separate selection cases. The fixed-layer
  and composed tests are **post-hoc exploratory follow-ups** on the same test
  data, not independent confirmations. Failed-question results are also saved.
- The qualitative gallery includes both original raw paper prompts and labelled
  chat adaptations. Raw inputs deliberately bypass chat templating. These are
  transfers to Gemma, not reproductions of the paper's larger-model results.
  The three-step example failed; its intermediates 21 and 42 had no accepted
  single-token representations, so their ranks were unavailable.
- Short-hop maps estimate **same-position** effects; the J/R baselines aggregate
  effects across **current and future positions**. Gemma 4 also has per-layer
  inputs and shared attention state. Thus the composition test is not a clean
  isolation of `E[BA]` versus `E[B]E[A]` for complete, compatible Jacobians.
  R/TR maps additionally use modified backward propagation, not literal derivatives.
- One corpus/sign seed and 32 fitting prompts cannot establish estimator
  stability. Better state reconstruction does not guarantee a better concept
  readout. None of these experiments demonstrates improved model reasoning or
  causal necessity of the labelled intermediates.

Cross-token transport was discussed as a possible future direction but **was not
implemented or tested**.

### Code and reproducibility

The fork adds [short-hop fitting and atlases](jlens/short_hop.py),
[activation means](jlens/stats.py), [Gemma R-style rules](jlens/relp.py),
the [arithmetic runner](experiments/short_hop/benchmark.py),
[fixed-layer analysis](experiments/short_hop/fixed_layer.py), and
[composed evaluation](experiments/short_hop/composition.py). Gemma 4 E4B support
includes shared-KV/forward-path checks and architecture-specific R/TR rules.

[RunPod utilities](scripts/runpod.py) save SSH host/port defaults, sync with
`--no-owner --no-group`, and launch detached jobs. Runs have overall/per-method
progress and interruption-safe checkpoints. The E4B full pilot ran on an A40;
the recorded invocation took approximately 76 minutes. Fixed-layer analyses
reused saved scores; composed evaluation reused fitted maps and answers but
reran forward passes to obtain activations.

Generated results live under `runs/` (gitignored, not bundled with this fork):

- `tjlens-a40/REPORT.md`: original lookup pilot.
- `gemma4-e4b/REPORT.md` and `qualitative/GALLERY.md`: full arithmetic run and gallery.
- `gemma4-e4b-layer25-tr/REPORT.md`: fixed layer-25 logit/R/TJ/TR comparison.
- `gemma4-e4b-composed25/REPORT.md`: composed TJ/TR with J/R baselines at 25 and 33.

Each run retains machine-readable metrics and provenance alongside its report.
See the [run guide](experiments/short_hop/README.md) to reproduce the workflows.
