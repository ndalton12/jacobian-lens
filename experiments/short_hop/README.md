# TJ-Lens: a short-hop signs-of-life experiment

This pilot asks whether short-hop Jacobian transport makes a known reasoning
intermediate easier to detect than regular J-Lens or R-Lens. TR-Lens applies
R-style backward rules to the same short-hop maps. The experiment does not change
the model's answers. Read `runs/<name>/REPORT.md` first when a run finishes.

The default experiment uses `google/gemma-3-1b-it`, one BF16 CUDA GPU, the spec's
16 maps for each of TJ and TR, 32 shared WikiText-103 fitting prompts, and 128 controlled two/three-hop
lookup cases. No quantization or compilation. Model dimensions come from config;
the target-layer schedule is specifically for this 26-block checkpoint.

## Gemma 4 E4B: fit and run the arithmetic comparison

For the E4B model that passed the practice check, use the new `--fit` path. It
fits fresh J/R/TJ/TR maps **only after** practice passes, then runs arithmetic and
the paper gallery. Old Gemma 3 maps are incompatible. Existing practice-only runs
from the earlier code have different source fingerprints: use a new run name;
the 32 cheap practice questions are rerun with the same default seed and task size.

```bash
uv run --no-project python scripts/runpod.py push

# First: real-GPU end-to-end smoke, detached and resumable.
uv run --no-project python scripts/runpod.py benchmark --name gemma4-e4b-smoke -- \
  --model google/gemma-4-E4B-it --fit --smoke
uv run --no-project python scripts/runpod.py logs --name gemma4-e4b-smoke

# Once the smoke finishes successfully (its science verdict is INCONCLUSIVE):
uv run --no-project python scripts/runpod.py benchmark --name gemma4-e4b -- \
  --model google/gemma-4-E4B-it --fit
uv run --no-project python scripts/runpod.py logs --name gemma4-e4b
uv run --no-project python scripts/runpod.py status --name gemma4-e4b
uv run --no-project python scripts/runpod.py pull
```

Direct GPU command:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.benchmark \
  --model google/gemma-4-E4B-it --fit --output-dir runs/gemma4-e4b
```

There are no new dependencies beyond `uv.lock`. The full run defaults to 32
shared fitting prompts of up to 64 tokens, dimension batches of **8**, and the
existing 192 arithmetic prompts. The smoke uses one fitting prompt capped at 16
tokens, sources 21/24 to target 25 (straddling the KV-sharing boundary), J/R to
layer 41, the same 32 practice questions, the fixed gallery, and eight main cases.
It is a compatibility check, not evidence of lens quality. Run the full experiment
in a different directory; smoke checkpoints are not compatible with full settings.

All generation and fitting work is resumable. Reissue the same command after an
interruption. Progress includes dimension batches within each fitting method and
evaluation cases. There is no one-hour deadline. **E4B fitting cost has not been
measured on an A40 here**; generating answers cheaply does not imply cheap Jacobian
fitting. Smoke measures viability, not a guaranteed full-run runtime. If smoke
runs out of memory, retry with `--dim-batch 4` and a new run name, and use that
batch size for the full run. Allow disk for the larger model plus several GB of
matrices/checkpoints.

If practice fails on a fresh-fit run, **no fitting or gallery is performed**:
there are no fitted maps yet. With `--reuse-from`, the existing behavior remains:
practice failure skips arithmetic but still produces the gallery from saved maps.
To re-evaluate E4B maps later, supply both `--model google/gemma-4-E4B-it` and
`--reuse-from runs/gemma4-e4b`, using a new output directory.

### Architecture and derivative conventions

E4B has 42 text blocks and residual width 2560. The four targets are **13, 23, 33,
41**, each with hops of **1, 2, 4, 8** blocks. These are zero-based block-output
indices; target 23 is the last block before the 18 KV-sharing blocks. J/R cover
every matching source and target the final block 41. The default target depths
scale with model depth; actual hop lengths stay fixed. No layer is chosen based
on test-case performance.

The Gemma 4 dense R/TR adaptation uses a separately versioned rule:
`gemma4-dense-relp-ln-identity-half-ple-v1`. Unlike Gemma 3, its RMSNorm scale is
`weight`, not `1 + weight`. We detach the RMS denominator in the four residual
norms and the post-per-layer-input norm; detach nonlinear activation factors in
the MLP and per-layer input gate; and split both gated products' gradients equally
between their factors. Attention, q/k/v norms, embedding construction, and layer
scalars retain their original behavior. MoE Gemma 4 variants are explicitly
rejected; this is the dense E4B/E2B rule, not a generic Gemma 4 MoE implementation.

The ordinary J/TJ maps differentiate a **residual-output intervention**, not all
model state. Token-derived per-layer inputs and KV computed before the source
output remain fixed; KV computed downstream remains connected to autograd and
contributes through every reuse. Consequently, a residual vector alone is not a
complete state for this architecture. The centered transport readout remains an
approximation and its state-prediction error is only a diagnostic. R/TR maps are
modified backward propagation, not literal Jacobians or published E4B results.

Before fitting, the real-checkpoint guard compares full HF logits against the
bare text path used by the fitter, checks a finite gradient across KV sharing,
verifies bit-exact R forward preservation, and checks restoration. Its results and
architecture metadata are saved in `benchmark_config.json` and fitting metadata.
The wrapper and bare decoder must produce bit-identical final residual states.
Their readouts use matching whole-sequence projection shapes and a dtype-aware
logit tolerance: BF16 rounding is not treated as a model-path failure. The actual
errors, dtype, and tolerance are recorded in the path-check metadata.

If an earlier run stopped with `bare text path differs from full HF logits
(max error 0.0625)`, push the fix and restart with a **new name**, for example
`--name gemma4-e4b-smoke-v2` (or `gemma4-e4b-v2` for the full run). The old guard
compared one-token and whole-sequence projections with a precision-inappropriate
tolerance. Code fingerprints intentionally prevent resuming that old directory
under changed code. This error occurred before fitting, so no completed fitting
work is lost; the new run repeats the practice check.

CPU tests use genuine tiny Gemma 4 modules with PLE and KV sharing, compare
derivatives with finite residual perturbations, detect deliberately detached KV,
and test the multimodal wrapper's text path, cache parity, FP32/BF16 forward
preservation, R corrections, and interrupted fitting/evaluation.

Optional real-checkpoint guard (downloads E4B; needs CUDA):

```bash
RUN_GEMMA4_TESTS=1 uv run --locked --extra dev --extra experiment \
  pytest -q tests/test_gemma4_integration.py
```

Source: [E4B configuration](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json).

### Fixed layer-25 follow-up (cached results, no GPU)

The full E4B run already saves logit lens at 25, R-Lens starting at 25 (mapping
to final layer 41), and TJ/TR predicted-state 25 → 33. Compare those exact readouts
on the same test questions without refitting, generation, subtraction, or layer
selection:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.fixed_layer \
  --run-dir runs/gemma4-e4b --output-dir runs/gemma4-e4b-layer25-tr
```

Outputs are `REPORT.md`, `summary.json`, and `cases.csv` in a separate, initially
empty directory; the original run is untouched. All methods must have every
test case. The headline uses solved cases; all-test comparisons are also saved.
Uncertainty resamples whole problem families, keeping paraphrases together.
This is an exploratory follow-up on previously inspected data, not a fresh
confirmatory test. Layer numbers are zero-based block-output indices; the token
position remains the final input token used by the original benchmark.

### Composed TJ/TR: 25 → 33 → final (forward-only GPU evaluation)

This is **different from the original TJ/TR readout**. The original short-hop
experiment applied the logit lens directly to the predicted layer-33 state. The
composed experiment adds the saved **J or R map from 33 to the final layer**
before unembedding. It includes J and R baselines at both actual layers 25 and
33, at the same final input token. Target-layer baselines use the model's actual
later state (more computation), not a predicted state.

Using column-vector notation, two variants are explicitly separated:

- **Matrix product:** TJ = `unembed(J_33 @ K_25_33 @ h_25)`;
  TR = `unembed(R_33 @ K_R_25_33 @ h_25)`.
- **Centered predicted state:** TJ =
  `unembed(J_33 @ (mean_33 + K_25_33 @ (h_25 - mean_25)))`, and likewise
  with R/TR maps for TR. Centering applies only to the first hop.

There is no normalization between maps. The final norm/LM head is applied once,
after the last map. **No change-only subtraction** is evaluated. Logit-lens and
old short-hop-only readouts are retained as secondary diagnostics.

```bash
uv run --no-project python scripts/runpod.py push
uv run --no-project python scripts/runpod.py compose -- \
  --reuse-from runs/gemma4-e4b
uv run --no-project python scripts/runpod.py logs --name gemma4-e4b-composed25
uv run --no-project python scripts/runpod.py status --name gemma4-e4b-composed25
uv run --no-project python scripts/runpod.py pull
```

Or directly on the GPU:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.composition \
  --reuse-from runs/gemma4-e4b --output-dir runs/gemma4-e4b-composed25
```

`--plan` validates the source case definitions and prints the plan without loading
a model. Defaults fix source 25 and target 33; no layer search. All 128 existing
test prompts are forwarded again (activations were not saved), but there is
**no refitting or answer generation**. Original correctness labels are retained;
the headline uses solved questions, with all-test comparisons also saved.
The original run and artifacts are read-only. A new `REPORT.md`, `summary.json`,
`cases.csv`, detailed per-alias/top-word readouts, and atomic per-case checkpoints
are written in the separate output directory. The detached RunPod job survives
SSH disconnects; rerun the same command to resume after interruption. Artifacts,
model revision, inputs and software are checked before reuse.

This probes the composition-of-averages idea, but it is **not a clean full
Jacobian chain-rule experiment**: short-hop maps estimate same-position effects,
whereas the J/R maps sum across future positions. Gemma also has shared KV and
per-layer input paths, and R/TR use modified backward rules. Product-vs-direct
differences therefore cannot be attributed only to averaging before composition.
This is an exploratory follow-up on already inspected data; replicate a positive
result on new cases and another fitting seed.

## New: arithmetic benchmark and paper-example gallery

After the lookup pilot, run this **evaluation-only** follow-up. It reuses all four
fitted artifacts from the existing run, leaving that run unchanged. No fitting or
new dependencies are required. Push the updated code, then launch on your pod:

```bash
uv run --no-project python scripts/runpod.py push
uv run --no-project python scripts/runpod.py benchmark -- \
  --reuse-from runs/tjlens-a40
uv run --no-project python scripts/runpod.py logs --name tjlens-arithmetic
uv run --no-project python scripts/runpod.py status --name tjlens-arithmetic
uv run --no-project python scripts/runpod.py pull
```

`benchmark` uses the same detached `nohup`/`flock` launch as `run`. Disconnecting
does not stop it. Reissue the identical command to resume; completed generation
and scoring cases are cached atomically. Artifacts, model revision, code/software,
datasets, and scoring settings are fingerprinted. Changes require a new output
directory/name. Do **not** rerun the old fitting command to start this benchmark.
The original `run` command remains available for fitting and the original lookup
experiment; its results are not overwritten or silently reinterpreted.

Direct GPU equivalent:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.benchmark \
  --reuse-from runs/tjlens-a40 --output-dir runs/tjlens-arithmetic
```

To check only model competence before doing any fitting, no saved lenses required:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.benchmark \
  --preflight-only --output-dir runs/arithmetic-practice
```

For a practice-only run that you intend to extend into a full comparison, supply
the same `--reuse-from` and output directory to both invocations, omitting just
`--preflight-only` on the second. `--plan` prints the plan without a model download.

### Questions, controls, and screening

The task is small-integer arithmetic, e.g. `(2 + 3) * 4` → **20**, with **5** as
the labelled intermediate. A paired problem such as `(6 - 2) * 5` has the same
answer but intermediate **4**. The benchmark asks whether each lens distinguishes
the internal steps rather than merely revealing the shared answer. An intermediate
is never one of that problem's explicit operands. Correct answers do not prove
that the model actually followed this computation; this is not a causal test.
The paired control number can itself occur as an operand in the other problem;
it is a different intermediate, not guaranteed to be completely irrelevant.

- First, 32 generation-only practice prompts must achieve **at least 80%** accuracy
  (26/32). Failure stops the main benchmark, not the fixed qualitative gallery.
  There is no automatic threshold relaxation or replacement of failed problems.
- Main benchmark: **192 prompts**, comprising 48 paired problem families, two
  problems per family and two wordings per problem (symbolic and verbal).
  There are 64 selection prompts and 128 test prompts. `--n-cases` may change the
  size; it must be at least 24 and divisible by 12.
- Practice families are disjoint from selection/test. Commuted addition duplicates
  are excluded. Both wordings and both same-answer controls stay in one split;
  bootstrap uncertainty resamples whole families, not individual paraphrases.
- Problem eligibility depends on fixed arithmetic rules and whole-token alias
  availability, **never** model answers or lens scores. The eligible final numbers
  and intermediates therefore depend on the tokenizer, and are recorded in JSONL.
- Numeric answers accept digit and English-word forms, case, surrounding whitespace,
  simple trailing punctuation, and integer-valued decimal forms such as `20.0`.
  Prose explanations and ambiguous responses such as `20 or 30` are not accepted.
- Intermediate/control/answer readouts use the best rank among predeclared complete
  single-token aliases (digits/words, capitalization, optional leading space).
  No partial number tokens, unknown tokens, or post-hoc synonyms are accepted.
  Per-alias ranks are retained. This alias-aware metric differs from the original
  lookup pilot's exact-token metric; don't directly compare the two headline scores.

The root `REPORT.md` explains ranks, spans, and **wins / ties / losses** in plain
language. It includes TJ/TR versus J/R, predicted-state versus change-only readouts,
wording sensitivity at frozen layer choices, same-answer controls, and separate
all-case/failed-case diagnostics. The main verdict still uses solved held-out cases
and the original pilot thresholds. Wordings aren't counted as independent evidence.
`progress.json` and logs show screening, gallery, each lens's evaluated-case count,
and report generation. Method progress advances together because readouts share one
forward pass. A failed preflight leaves the main-method progress uncompleted.

### Fixed qualitative comparisons

`qualitative/GALLERY.md` shows model completions (including failures), top-word
tables and intermediate-rank plots for **J, R, TJ predicted state, TJ change-only,
TR predicted state and TR change-only**. Tables use the same structurally chosen
source/target pair for all methods, not the best-looking pair per example; full
readouts for every fitted pair are also saved. Curves distinguish source-layer
baselines from target-layer transport, with separate hop-length curves.

The examples are fixed in code before examining results:

- Fourth planet → Mars → red (`lens-eval-multihop.json`, `mars-color`).
- `(2 + 3) * 4` → 5 → 20 (`lens-eval-order-ops.json`, `parens-add-mult`).
- `(4 + 17) * 2 + 7` → 21 → 42 → 49 (paper Figure 17).
- A sentence ending in `langauge` → language (`lens-eval-typo.json`, `typo-language`).

Each includes the exact upstream raw prompt at its final input token and a
**separately labelled chat adaptation**. Original raw completions allow further
text after the expected answer; raw typo recognition has no answer-accuracy test.
If a whole intermediate such as 21 or 42 has no single-token alias, the gallery
explicitly marks its rank unavailable, but still shows top words. It never treats
the first digit as the entire number. Two tracked intermediates in the long example
produce ten case records across the four examples and two input formats.

These are qualitative transfers to Gemma, not reproductions of the larger-model
results. The gallery is excluded from layer selection and the arithmetic verdict.
Sources: [upstream evaluation conventions](../../data/evaluations/README.md) and
[J-Lens paper](https://transformer-circuits.pub/2026/workspace/index.html).

Output layout:

```text
runs/tjlens-arithmetic/
  REPORT.md, summary.json, progress.json, benchmark_config.json
  practice_cases.jsonl, arithmetic_cases.jsonl, gallery_cases.jsonl
  preflight/       # resumable practice answers, accuracy and gate decision
  arithmetic/     # all four methods, detailed metrics, transactions and plots
  qualitative/    # GALLERY.md, rank plots, all readouts and case transactions
```

## Run on an existing RunPod

Choose an A40 pod with direct SSH-over-TCP access and a persistent `/workspace`.
The RunPod proxy SSH endpoint does not support rsync; use its direct IP and mapped
SSH port. See [RunPod's SSH guide](https://docs.runpod.io/pods/configuration/use-ssh).
CUDA **13.x is supported; it need not be exactly 13.0**. The lockfile includes
the CUDA 13.0 runtime dependencies for PyTorch on Linux. A pod with a driver
supporting CUDA 13.1, 13.2, or later can run that older runtime through backward
compatibility. NVIDIA lists driver branch **580 or newer** for CUDA 13.x.
The container's installed toolkit and `nvidia-smi`'s advertised CUDA capability
need not equal `torch.version.cuda`, which describes the runtime used by PyTorch.
Installing a different container cannot upgrade the host driver.

`setup` and model startup execute a real BF16 CUDA matrix multiply,
normalization, and backward pass, and print the GPU, driver, and PyTorch CUDA
runtime. This verifies the environment before the expensive model fit. You can
run the same check with:

```bash
uv run --locked python -m experiments.short_hop.check_gpu
```

See NVIDIA's [backward compatibility explanation](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html)
and [CUDA 13.x driver table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
For newer runtimes on older drivers, minor-version compatibility has limitations
around new features and PTX; using the locked runtime with a newer driver avoids
that particular mismatch.

All commands run from the repository root. The local utility uses only the
standard library, so it does not install PyTorch on your laptop.

```bash
# Save once; configure again when the pod's address changes.
uv run --no-project python scripts/runpod.py configure \
  --host YOUR_POD_IP --port YOUR_SSH_PORT --user root \
  --identity /absolute/path/to/your/ssh_key

uv run --no-project python scripts/runpod.py push
uv run --no-project python scripts/runpod.py setup

# Accept Gemma's license on Hugging Face, then authenticate on the pod.
# Omit this if HF_TOKEN is already supplied through the pod environment.
uv run --no-project python scripts/runpod.py exec uv run hf auth login

# Run the small GPU check first; it runs detached from your terminal.
uv run --no-project python scripts/runpod.py smoke
uv run --no-project python scripts/runpod.py logs --name tjlens-smoke

# After smoke completes successfully:
uv run --no-project python scripts/runpod.py run
uv run --no-project python scripts/runpod.py logs
uv run --no-project python scripts/runpod.py status
uv run --no-project python scripts/runpod.py pull
```

Defaults live in gitignored `.runpod.json`; the identity value is a path, never
the key contents. Both push and pull pass `--no-owner --no-group --no-perms`.
Sync excludes virtual environments, git metadata, run outputs when pushing,
`.env*`, private-key file extensions, and local SSH configuration. It never uses
`--delete`. Pull copies remote `runs/` into local `runs/`, updating same-named files.
Use unique run names to retain separate experiments. Normal SSH host-key checking
is preserved. `--dry-run` before the subcommand prints commands without connecting.

`setup` installs uv if needed using its official installer, then performs
`uv sync --locked --extra dev --extra experiment` and CPU tests. It requires
Python >=3.10, curl, SSH/rsync, and `flock` (standard on Linux pods). Dependencies
and model downloads are a separate first-time setup cost. Provision enough disk
for the CUDA wheels, caches, model weights, and checkpoints (roughly 20 GB free
is a reasonable starting allowance; actual usage varies).

Run management covers an existing SSH-accessible pod: it does not provision,
bill for, stop, or terminate pods. `status` shows the GPU and progress summary. After
collecting results, stop the pod in RunPod to stop paying for GPU time.

## Run directly on the GPU

```bash
uv sync --locked --extra dev --extra experiment
uv run --locked pytest -q
uv run --locked --extra experiment python -m experiments.short_hop.run --plan
uv run --locked --extra experiment python -m experiments.short_hop.run \
  --smoke --output-dir runs/tjlens-smoke
uv run --locked --extra experiment python -m experiments.short_hop.run \
  --output-dir runs/tjlens-a40
```

All four methods now run to completion on the same fixed 32-prompt corpus by
default. There is no automatic one-hour cutoff or reduction in fitting data.
Use `--n-prompts N` to choose another common count. An optional `--minutes N`
sets a deadline for that invocation without changing the dataset. Deadlines
are checked between fitting prompts and evaluation cases; a single operation
or initial download can overrun the deadline. No A40 runtime measurement has
been made locally.

If a run times out, crashes, or is interrupted, `REPORT.md` says incomplete or
inconclusive and retains checkpoints. Reissue the same command to resume fitting
**and evaluation**. Fitting checkpoints are saved atomically after each prompt,
separately for each method and target. Completed evaluation cases are saved as
atomic transactions in `evaluation_cases/`; generation and scoring are skipped
for those cases on resume. Consolidated JSONL files are rebuilt from the saved
transactions, so interrupted writes do not duplicate rows. At most the current
unsaved fitting prompt or evaluation case must be repeated. Shared means are
also cached. SIGTERM and Ctrl-C mark the run interrupted; a hard kill still leaves
the last atomic checkpoints available. Keep the run directory on persistent pod
storage. You can increase or omit `--minutes` on resume.
Changes to model, data, seeds, code/dependency provenance, or fitting settings
require a new output directory. A completed run prints its existing report.
On OOM, use a new run with `--dim-batch 8`. `--dim-batch 32` is available if memory
allows, but the default is 16.

Use a new directory when moving from the earlier three-method runner to this
four-method version; the format/configuration check prevents mixing experiments.

For an optional two-hour deadline:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.run \
  --output-dir runs/tjlens-full32 --n-prompts 32 --minutes 120

# Equivalent remote invocation:
uv run --no-project python scripts/runpod.py run --name tjlens-full32 -- \
  --n-prompts 32 --minutes 120
```

## Progress tracking

`run.log` shows progress through each method's targets, fitting prompts, and
backward batches. For Gemma's 1,152 dimensions with `dim_batch=16`, each fitting
prompt has 72 backward batches. A typical line is:

```text
Overall ~63.4% | TR-Lens 28.1% | TR-Lens/target_14 | prompt 5/32, backward batch 36/72
```

`progress.json` is updated atomically about every two seconds while backward
batches advance; log updates are throttled to about ten seconds and prompt/job
boundaries. It includes per-method percentages, overall progress, current work,
PID, last update time, and elapsed time for the invocation. Overall percentage
is an estimate of work (fitting is weighted by layer span), not a promised ETA.
Model loading appears as the starting phase. Percentages are reconstructed from
saved work after a restart.

```bash
uv run --no-project python scripts/runpod.py status --name tjlens-a40
uv run --no-project python scripts/runpod.py logs --name tjlens-a40
# Directly on the pod:
uv run --locked python -m experiments.short_hop.progress --run-dir runs/tjlens-a40
```

If the process was killed before it could update the snapshot, `status` detects
that the recorded PID is gone. Running the same `run` command resumes the job;
locks prevent two runners from writing the same directory concurrently.

## What the result means

`REPORT.md` says `PROMISING`, `MIXED`, `NO_CLEAR_IMPROVEMENT`, or `INCONCLUSIVE`.
It gives the intermediate-rank effect size versus J-Lens and R-Lens, a confidence
interval, case win rate, task-specific results, query relevance, and target-state
prediction quality. It gives separate TJ and TR verdicts and a TR-versus-TJ
comparison. `summary.json` contains the exact decisions and thresholds, including
TR results under `tr_lens`. Both select their spans on the selection split only.

The primary TJ score is the spec's **difference of complete readouts**:

```text
x_i = h_i - mean_i
transported = unembed(mean_j + x_i @ K.T)
identity    = unembed(mean_j + x_i)
innovation  = transported - identity
```

RMSNorm is applied to complete states. We never unembed a standalone innovation
vector to get the primary score. FP32 is used for Jacobian accumulation, means,
transport, logit differences, and metrics; the model and its unembedding run in
BF16. Matrices are serialized as FP16 and reloaded before evaluation. FP32 means
are retained. The optional norm-free linear innovation diagnostic is omitted.

Every case records nine readouts: TJ innovation, TJ centered transported state,
centered identity, actual target logit lens, source logit lens, J-Lens, and
R-Lens, plus `tr_innovation` and `tr_transported`.
TJ/TR share exactly the same means, source states, position estimator, sign seed,
layer pairs, and fitting prompts. Only their backward rules differ. Generation
and the activation recording pass are shared across all four methods per case.
TR retains centered transport and the difference-of-complete-readouts score.
Its maps are modified propagation coefficients rather than literal Jacobians;
state prediction is a diagnostic for both methods.
J/R use the upstream uncentered `unembed(h_i @ J.T)` with `future_sum`.
Both have the same corpus, prompt count, valid-position mask, source layers,
and final-block target. The atlas uses `self_hutchinson` by default. This bundles
several method differences; a win does not isolate which change caused it.

Half the fact groups select layers; the other half test the frozen choices.
Each method independently picks its best source/span by mean log2 intermediate
rank on solved selection cases. TJ's headline uses innovation; the best
transported-state span is selected and reported separately. Baselines are not
restricted to a poor layer chosen for TJ. Additional comparisons at TJ's selected
source are saved in `summary.json`. The bootstrap resamples whole fact groups,
preserving paired-query dependence; it is conditional on this fitted seed.

A meaningful pilot advantage requires at least a **2x improvement in geometric
mean intermediate rank**, a positive 95% bootstrap lower bound in log2 rank gain,
and lower ranks on at least 60% of test cases. At least 8 fitting prompts, 24
solved test cases, and 12 fact groups are required. A `PROMISING` verdict also
requires improvement over R-Lens and that the relevant intermediate outrank its
paired irrelevant label on more than half the test cases. These thresholds are
predeclared practical heuristics, not claims of general scientific validity.

## Tasks and limitations

There are 64 two-hop and 64 three-hop cases by default; `--hops 2` makes all
cases two-hop. Each fact group produces two questions with identical shuffled
facts and different queried nonce entities. The other question's first-hop
label acts as a relevance control. Half the groups add two distractor chains.
Group pairs stay in the same selection/test split, and splits balance hop counts.
Three-hop metadata retains both intermediates; the primary single-token target
is the **first-hop intermediate** in both task types. No world knowledge is needed.

Candidate label pools are filtered for single-token continuations at the actual
assistant boundary, then checked again for every case. The model receives chat
template token IDs directly, without double tokenization. Greedy generation must
match the answer, allowing capitalization and surrounding periods/quotes. A
reasoning preamble is counted as unsolved under the one-word-response instruction.
All cases are retained; headline lens comparisons use only solved cases.

This is an association/readout test, not a causal-intervention experiment. The
intermediate labels occur explicitly in the facts. Query-switch controls help
distinguish relevant readout from generic copying, but do not prove the model
internally used a particular intermediate. Few solved cases on the 1B model
produce an inconclusive result. Inspect task-specific solve rates before scaling
up. A positive finding needs a second corpus/sign seed and stronger controls.

The R-Lens implementation follows the published dense-model LN, activation
identity, and gated-product half rules. It changes only residual RMSNorms and
MLPs; attention and q/k norms are untouched. The patch is reversible and tested
for identical forward outputs plus changed gradients. The runner checks exact
forward equality on a fitting prompt before accepting the baseline.

Sources: [R-Lens method](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers),
[authors' artifacts and recipe](https://huggingface.co/camilablank/workspace-lenses),
[Gemma implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py),
[upstream J-Lens paper](https://transformer-circuits.pub/2026/workspace/index.html).

## Changes and ambiguities relative to the supplied MVP

| Topic | Implemented choice |
| --- | --- |
| Fixed 32 prompts vs roughly one hour | Fixed 32 for all four methods, no default cutoff, per the updated request. |
| Compare only identity/local vs compare J/R | Add separately fitted final-layer J/R baselines; never reuse a same-position TJ matrix as regular J-Lens. |
| TR comparison | Fit the same short hops with R-style rules; compare TR with TJ, J, and R on the same held-out data. |
| Released R-Lens recipe | Rules adapted to Gemma 3 1B. Published artifacts used a penultimate target, 25 Pile prompts and skip=4; this pilot matches upstream default final-target J-Lens with WikiText and skip=8. This is not a published-score reproduction. |
| 64 two-hop examples vs stronger comparison | Default 128 two/three-hop cases, separate selection/test groups, query-switch controls. |
| Best layer as a result | Choose on selection cases, evaluate on held-out cases; plots over all test layers are exploratory. |
| Innovation vs state prediction | Headline intermediate ranks use innovation; transported-state comparison and state fidelity are reported separately. |
| Future-sum ablation / corpus-seed stability | Optional follow-up runs. Do not infer stability from one seed. |

To compare future-sum transport on the same deterministic dataset, use a new run
with `--position-reduction future_sum --n-prompts N`, where `N` is the first run's
chosen prompt count. Use `--seed 1` in another run to check seed sensitivity. Those
are additional experiments beyond the default comparison.

## Artifacts and individual commands

Each run writes `config.json`, the fitting/evaluation JSONL datasets, per-target
and baseline checkpoints, `short_hop_atlas.pt`, `tr_short_hop_atlas.pt`, `j_lens.pt`, `r_lens.pt`,
`generation_results.jsonl`, `pair_results.jsonl`, `aggregate_metrics.csv`,
`summary.json`, `REPORT.md`, `progress.json`, resumable `evaluation_cases/`, and
four PNG plots for TJ plus four in `plots/tr_lens/`. Checkpoints for the atlases
live separately under `checkpoints/tj/` and `checkpoints/tr/`. Provenance includes exact corpus
and case hashes, source-code hash, model revision, library versions, GPU, seeds,
layer mapping, estimator, and chosen fitting count. `config.json` status must be
`complete` for a finished experiment. Pair results include full-vocabulary ranks
and top-20 tokens, never complete vocabulary logits.

The modular commands from the spec are also available. Each supports `--help`:

```bash
uv run --locked --extra experiment python -m experiments.short_hop.build_fit_corpus \
  --output runs/manual/fit_prompts.jsonl --n-prompts 32 --seq-len 64
uv run --locked --extra experiment python -m experiments.short_hop.build_eval_cases \
  --output runs/manual/eval_cases.jsonl --n-cases 128
uv run --locked --extra experiment python -m experiments.short_hop.fit_atlas \
  --prompts runs/manual/fit_prompts.jsonl --output runs/manual/short_hop_atlas.pt \
  --checkpoint-dir runs/manual/checkpoints
# evaluate requires separately fitted J/R artifacts; the one-command runner creates them.
uv run --locked --extra experiment python -m experiments.short_hop.evaluate \
  --atlas runs/tjlens-a40/short_hop_atlas.pt --j-lens runs/tjlens-a40/j_lens.pt \
  --tr-atlas runs/tjlens-a40/tr_short_hop_atlas.pt \
  --r-lens runs/tjlens-a40/r_lens.pt --cases runs/tjlens-a40/eval_cases.jsonl \
  --output-dir runs/tjlens-reeval
uv run --locked --extra experiment python -m experiments.short_hop.plot_results \
  --results runs/tjlens-a40/pair_results.jsonl --output-dir runs/tjlens-a40/plots
```

CPU verification and optional authenticated GPU integration:

```bash
uv run --locked --extra dev --extra experiment pytest -q
uv run --locked --extra dev ruff check .
RUN_GEMMA_TESTS=1 uv run --locked --extra dev --extra experiment \
  pytest -q tests/test_gemma_short_hop_integration.py
```

The gated test additionally requires `HF_TOKEN` and CUDA. Tiny CPU Gemma tests use
random small models instantiated from config; they do not download gated weights.
