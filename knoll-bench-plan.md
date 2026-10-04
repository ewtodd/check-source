# Knoll Bench — Specification and Generation Plan

Working name: Knoll Bench, after Glenn F. Knoll's *Radiation Detection and Measurement*.

Status: draft for review, 2026-10-03.
Scope: spec, formats, source-mining and verification workflow. No implementation here.
Context: extends the `gsm8k-eval` harness (same endpoint/client style) to a private
benchmark for experimental physics data analysis, generated from already completed and
verified analyses in UM-ANSG and, locally, the MUSIC repositories.

---

## 0. Purpose and success criteria

The benchmark exists to answer one question repeatedly: **which quantization/checkpoint
of a given base model is best for the work I actually do**, without the answer depending
on the runner.

Success criteria:

1. Two checkpoints of the same base model can be compared on identical items with a
   paired statistic whose uncertainty is smaller than the effect being hunted.
2. The same item set can be run under Runner A (single-turn, now) and Runner B
   (Autophysicist, later), and the per-item scores are directly comparable.
3. Every gold answer is traceable to a committed artifact, a published value, or an
   executed reference computation. No check's gold value is ever typed from a model's
   memory.
4. The item set is private by default, versioned by hash, and split into a development
   half and a held-out half that is not used while authoring prompts.

Non-goals:

- A public leaderboard.
- Replacing GSM8K or lm-eval; the existing harness and its AMD calibration numbers stay
  valid and untouched.
- LLM-judge scoring in the core set. If a rubric-scored category is added later it lives
  in a separate, clearly labeled section and never gates the primary comparison.

---

## 1. Design principles

1. **Gold comes from artifacts, not assertions.** A generator may write locators,
   prompts and schema fields. A builder resolves locators against files and fills in
   numbers. If the model's text and the artifact disagree, the artifact wins and the
   item fails validation.
2. **The runner is a variable, not part of the item.** Items define the task, the
   inputs, the output contract and the checks. Nothing in an item references
   Autophysicist, a particular prompt scaffold, or a particular sampler profile except
   through runner-neutral optional hints.
3. **Objective, deterministic scoring first.** Numeric checks against executed outputs,
   code against tests, fixed choice. Free text without a computable answer is out.
4. **Ambiguity is worse than a smaller set.** An item whose gold depends on an unstated
   convention (fit range, weighting, unit, calibration order) is rejected, not
   hand-waved with a loose tolerance.
5. **Code and reasoning mix naturally.** Items are tagged by task type and domain; the
   coding/reasoning split is a property, not a top-level partition.
6. **The benchmark measures models, the harness is calibrated separately.** Runner B
   results are never a gold source. The Autophysicist is under active repair
   (`son-of-anton/STATUS.md` item 9), so Runner A is the primary source of model
   comparisons until Runner B passes its qualification suite.

---

## 2. Layers and interfaces

### 2.1 Definitions

| Term | Meaning |
|---|---|
| Item | One frozen, independently scorable task: prompt, inputs, output contract, checks. |
| Check | One key/value obligation with a type, expected value, tolerance and gold reference. |
| Item set | A named, hashed collection of items plus a manifest. `knoll-fast-v1`, `knoll-full-v1`. |
| Runner | Executes items against a model endpoint and captures raw outputs and artifacts. |
| Scorer | Reads raw outputs and artifacts, applies checks, emits per-check results. |
| Executor | Runs model-generated code/patches in a sandbox before scoring (code items only). |
| Builder | Resolves gold locators, validates schema and cross-references, emits `items.jsonl`. |
| Generator | The local model performing the extraction pass described in section 6. |
| Gold reference | A machine-resolvable pointer to where a check's expected value comes from. |

### 2.2 Interface contract between runners and scorer

Both runners produce the same three artifacts per run:

```
run.json        environment and provenance (model, revision, args, seeds, hashes)
items.jsonl     one row per item: raw output, artifacts, extracted values, timing
summary.json    aggregate and per-category scores
```

The scorer reads `items.jsonl` and the item files and is runner-agnostic. A Runner B
result that cannot be normalized into this shape is not comparable and is discarded.

### 2.3 Factorial design

- Model comparison: item set frozen, runner pinned, only the checkpoint varies.
- Runner comparison: item set frozen, checkpoint pinned, Runner A vs Runner B.
- Interaction: the interesting quantity is `score_B(model) - score_A(model)`; if the
  Autophysicist scaffold helps a model, that delta should be positive and larger for
  stronger checkpoints. Report the delta per category, not just overall.

### 2.4 Runner A vs Runner B scope

- Runner A now: single-turn generation plus optional one-shot code execution in a
  sandbox. Covers most coding and reasoning items.
- Runner B later: the full agent loop; covers `tool_loop` and `agentic` horizon items
  (multi-step data analysis, intermediate files, iteration).
- Every item declares `horizon`; suites declare which horizons they contain. Item sets
  used for model-vs-harness comparisons contain the same items in both runners.

---

## 3. Source material

### 3.1 Source classes

| Class | Examples | Use |
|---|---|---|
| Verified analysis repos | UM-ANSG projects; MUSIC analyses (local only) | Primary task and gold source |
| Shared infrastructure | `Analysis-Utilities`, `analysis-caen`, `musicsim`, `nds-mcp` | Coding tasks, gold computation, data access |
| Committed outputs | `.result` tables, `.roofits`, ROOT files, PNG/PDF figures | Gold locators |
| Publications | 73mGe, 78mBr, YAP-PSD papers | Highest-trust gold; also difficulty statements |
| Datasets | `/labdata/ANSG/...`, private source files | Inputs; contamination-proof |
| Agent run records | `workspace-soa/runs/...` | Task ideas and difficulty notes only; never gold |
| Docs and method notes | `doc/pages/*.md`, Doxygen, README | Prompt specs, conventions |

### 3.2 Verified UM-ANSG inventory (inspected)

- `73mGe` — position-sensitive CZT. Macros: `ConvertBEF.cpp`, `Filter.cpp`,
  `PixelCalibration.cpp`, `CalibrationLow.cpp`, `AdditionalLevels.cpp`,
  `CombineGeResult.cpp`, `ExportForAlex.cpp`, plus `include/BEF.hpp`, `src/BEF.cpp`.
  Committed artifacts: `results/ge_amxfer.result`, `root_files/**`,
  `plots/calibration_low_amxfer/fits/*.roofits`, `plots/pixel_calibration/**`.
  Published final value: 68.7502 +/- 0.0142 keV (commit `c260617` records the result).
- `YAP-PSD` — published NIMA paper. Macros: `InitialProcessing.cpp`, `Calibration.cpp`,
  `ChargeComparison.cpp`, `ShapeIndicator.cpp`, `GateOptimization.cpp`, `NoiseRMS.cpp`,
  `AverageWaveforms.cpp`; Python: `analysis.py`, `psd_utils.py`, `regressors.py`,
  `torch_models.py`, `edge_study.py`, `noise_study.py`, `parameter_study.py`,
  `shuffle_study.py`, `proof_of_concept.py`; many committed plots.
- `78mBr` — published Nuclear Physics A paper. Macros: `InitialProcessing.cpp`,
  `Calibration.cpp`, `HalfLife.cpp`, `MonteCarloEfficiency.cpp`.
- `YAG-PreProcessing`, `ANSG-Archival` (EJ309B, SiDiode, BulkYAG, AuFoil, YAP),
  `ANSG-WIP/YAP-QDA` (work in progress; excluded until it has committed results).
- Unified flake with one dev shell per project
  (`nix develop .#73mGe`, `.#78mBr`, `.#YAG-PreProcessing`, `.#YAP-PSD`), pinned to one
  `Analysis-Utilities` and one ROOT. This is the reproducibility substrate for
  `recomputed` gold.

### 3.3 Restricted sources

MUSIC repositories and `/labdata` are DOE/lab sensitive. Rules:

- The generator that touches these sources must be a local model; restricted content is
  never sent to a remote or online endpoint.
- Items derived from restricted sources carry `sensitivity: restricted` and live in a
  local overlay (`items-local/`) that is never committed, pushed, or shared.
- Paths in prompts use placeholders (`${LABDATA}`, `${UM_ANSG}`, `${MUSIC_SRC}`)
  resolved from a local env file. The resolved paths never enter the item file.
- If a restricted item can be re-expressed on synthetic or published data with the same
  skill tested, that public variant is preferred for the shareable suite.

### 3.4 Data policy

Every item's inputs are either:

1. a repo artifact pinned by commit (preferred; small),
2. a read-only dataset pinned by path + SHA256 + tree name (for ROOT files), or
3. a small derived fixture with an explicit generation command and seed.

Large ROOT files are not copied into the benchmark repo. The manifest records the hash
of each input at the time the gold was established. Data drift (file replaced) is a
validation failure, not a silent retest.

---

## 4. Taxonomy

Top-level domains, drawn from the verified UM-ANSG and MUSIC work. The local model
confirms and extends this during the inventory pass; categories are stable once v1 is
frozen.

| Code | Domain | Typical task |
|---|---|---|
| CAL | Energy/gain/pixel calibration | Fit peaks, build calibration curves, transfer calibrations |
| SPEC | Spectroscopy and peak analysis | Identify lines, background subtraction, limits, efficiencies |
| PSD | Pulse-shape discrimination and waveforms | Features, classifiers, gate selection, ROC/AUC |
| NUC | Nuclear data | ENSDF/EGAF attribution, half-lives, branching, activation |
| SIM | Simulation and transport | MUSIC/LISE++/SRIM/Geant4 setups, efficiency, trace comparison |
| STAT | Statistics and fitting | Weighted means, uncertainty combination, chi2, covariance |
| SW | Software/data plumbing | ROOT macros, Analysis-Utilities, converters, Nix builds |
| DAQ | Acquisition/processing | Time-sort, filter, baseline, pile-up, polarity |

Task types (orthogonal to domain):

| Type | Description | Horizon |
|---|---|---|
| `compute` | Numeric answer from data or nuclear data, no code delivery required | single_turn |
| `interpret` | Choose/diagnose from provided results, produce keys | single_turn |
| `implement` | Write a function/module/macro matching a spec | single_turn or tool_loop |
| `script` | Write an analysis program that consumes inputs and writes `RESULTS.txt` | tool_loop |
| `debug` | Find and fix a fault in provided analysis code | tool_loop |
| `pipeline` | Multi-stage analysis with intermediate artifacts | agentic |

Every item is tagged `klass: coding | reasoning | mixed` derived from type and domain.
The user-visible split requested ("coding and reasoning problems") is a reporting view
over these tags, not a structural constraint on the files.

---

## 5. Formats

### 5.1 Repository layout

```
knoll-bench/
  PLAN.md
  README.md
  flake.nix / flake.lock          # harness environment, not analysis environments
  taxonomy.toml
  schema/
    item.schema.json
    check.schema.json
    inventory.schema.json
    run.schema.json
  sources.env.example             # placeholder -> local path map
  inventory/
    um-ansg.inventory.toml
    music.inventory.toml          # local only, never committed if restricted
  items/
    dev/                          # development split (prompt authoring allowed)
    test/                         # held-out split (frozen, local only if restricted)
  items-local/                    # restricted overlay, gitignored
  build/
    items.jsonl                   # generated, hash-stamped
    manifest.json
  generators/
    inventory.md                  # generator contract prompts (appendix B)
    extract.md
  tools/
    build.py
    validate.py
  runners/
    runner_a.py
    sandbox.py
    adapters/runner_b.py
  scores/
    scorer.py
    compare.py
  results/                        # gitignored
```

Items are one TOML file each. `build.py` compiles them to JSONL for runners and emits
`manifest.json` with per-item hashes and the set hash. TOML is used for every authored
specification (items, checks, inventory, taxonomy); JSON/JSONL stays for machine
interchange, because TOML has no line-delimited streaming form. Runners never read TOML
directly; scorers read raw output plus the compiled items.

### 5.2 Item format (source of truth, TOML)

```toml
schema_version = 1
id = "ansg-73mge-cal-low-insitu-cu10-a"
title = "Fit the 68.75 keV line in the Cu-shield in-situ spectrum"
klass = "reasoning"              # coding | reasoning | mixed
domain = "CAL"
type = "compute"                 # compute | interpret | implement | script | debug | pipeline
horizon = "single_turn"
est_runtime_s = 120
difficulty = 3                   # initial estimate 1-5, replaced by calibration data
sensitivity = "internal"         # public | internal | restricted
publishable = true

[source]
repo = "UM-ANSG"
commit = "9fc4510"
paths = [
  "73mGe/macros/CalibrationLow.cpp",
  "73mGe/root_files/calibrated_low_amxfer/CuShieldSignal_10Percent_20260113.root",
]

[[inputs]]
path = "work/CuShieldSignal_10Percent_20260113.root"
from = "${UM_ANSG}/73mGe/root_files/calibrated_low_amxfer/CuShieldSignal_10Percent_20260113.root"
sha256 = "<filled by builder>"
tree = ""
branches = []

[env]
nix_develop = "UM-ANSG#73mGe"
network = false
timeout_s = 900

prompt = '''
The mounted file work/CuShieldSignal_10Percent_20260113.root is a calibrated
CZT spectrum from the UM-ANSG 73mGe campaign (Cu shield, 10% setting, 2026-01-13).
The 68.75 keV isomeric line is the only structure in the fit window
60-78 keV; the background is linear.

Fit the line with a Gaussian plus a linear background over 60-78 keV and report,
one per line, `key = value` in RESULTS.txt:

- line_centroid_keV: centroid of the Gaussian, in keV
- line_sigma_keV: Gaussian sigma, in keV
- line_area: fitted net peak area, in counts
- chi2_per_ndf: chi2 divided by the number of degrees of freedom
'''

[output]
contract = "results_txt"
extraction = "results_txt_v1"
keys = ["line_centroid_keV", "line_sigma_keV", "line_area", "chi2_per_ndf"]

[[checks]]
id = "centroid"
key = "line_centroid_keV"
type = "numeric"
expected = "<resolved at build>"
units = "keV"
required = true

[checks.tol]
abs = 0.05

[checks.gold_ref]
kind = "committed_result"
repo = "UM-ANSG"
commit = "9fc4510"
file = "73mGe/results/ge_amxfer.result"
locator = 'label="Cu_Shield_Signal_10%_(01/13)" column=ge_mu'

[[checks]]
id = "sigma"
key = "line_sigma_keV"
type = "range"
lo = 0.1
hi = 5.0
units = "keV"
required = true

[checks.gold_ref]
kind = "manual"
note = "order-of-magnitude sanity bound; fit parameter depends on background choice"

[[checks]]
id = "chi2"
key = "chi2_per_ndf"
type = "range"
lo = 0.5
hi = 2.5
required = true

[checks.gold_ref]
kind = "manual"
note = "fit quality sanity bound, not a physics value"

[scoring]
mode = "partial"                # partial | all_or_nothing
pass_threshold = 1.0            # item "pass" needs every required check

[verification]
state = "draft"
machine = []

[verification.human]

[generation]
method = "g1_verified_analysis"
generator_model = "local"
generated_at = "2026-10-03"
seed = 20261003
notes = ""
```

### 5.3 Check format

Required fields per check: `id`, `key`, `type`, `required`, `gold_ref`.

Types:

| Type | Semantics |
|---|---|
| `numeric` | Parse a float; pass when `abs(got - expected) <= max(tol.abs, tol.rel * abs(expected))`. |
| `integer` | Exact integer equality after parsing. |
| `boolean` | Accept `true/false/yes/no/1/0` case-insensitively. |
| `string` | Equality after strip, whitespace collapse, optional case fold. |
| `enum` | Membership in `allowed`, with `aliases` map. |
| `range` | `lo <= got <= hi`; used for sanity bounds and for quantities whose exact value depends on a convention. |
| `exists` | Key present and parseable; no expected value. |
| `expression` | `sympy` equality of the parsed answer with `expected` (optional, later). |

Rules:

- `units` is mandatory for physical quantities and is used only for documentation and
  optional suffix stripping, never for silent conversion. If conversion is allowed,
  `accept_units` lists the accepted strings and the conversion is explicit.
- A `numeric` check with `rel` unset and a tolerance wider than 10% of the expected
  value must carry a human note explaining the physical uncertainty.
- Every check must be listed in `output.keys`; extra keys in `RESULTS.txt` are ignored.
- An item may have at most one `range` check per quantity; range-only items must be
  labeled `type: interpret` and are not used as the sole difficulty driver.

### 5.4 Gold references

Gold kinds, in decreasing trust:

| Kind | Builder behavior |
|---|---|
| `published` | Value must be present and is compared against committed artifacts if they exist. Requires human sign-off. |
| `committed_result` | Builder re-reads the file at the pinned commit, applies the locator, and asserts the stored `expected` matches. |
| `recomputed` | Builder runs the named script in the pinned environment and extracts the value. Failure rejects the item. |
| `derived` | Builder evaluates a small formula over other check values (weighted mean, propagation). Formula and inputs are explicit. |
| `manual` | Human-entered bound or convention-dependent value; flagged in the review queue. Never used for a headline result. |

Locator language (kept deliberately small): a file path plus one of

- `row label=<string> column=<header>` for whitespace/CSV tables,
- `regex: <pattern> group=<name>` for text and ROOT `TTree` dumps,
- `hist: <name> bin=<n>` for ROOT objects (via the analysis environment),
- `json: <pointer>` for JSON files.

The **number is never stored by the generator**; it is written by the builder after the
locator resolves. If a human edits a gold value by hand, the check is marked `manual`
and re-reviewed.

### 5.5 Runner outputs

`run.json`:

```json
{
  "runner": "runner_a",
  "runner_version": "0.1.0",
  "runner_git": "<hash>",
  "item_set": "knoll-fast-v1",
  "item_set_hash": "sha256:...",
  "model": "vllm/qwen3.8-27b",
  "served_name": "qwen3.8-27b",
  "endpoint": "http://10.0.0.6:4002/v1",
  "api": "chat",
  "sampler": {"temperature": 0.0, "top_p": 1.0, "seed": 1234, "max_tokens": 4096},
  "started_at": "...",
  "finished_at": "...",
  "host": "e-desktop",
  "nix_flake_lock": "<sha or store path>"
}
```

`items.jsonl` one object per item:

```json
{
  "item_id": "ansg-73mge-cal-low-insitu-cu10-a",
  "item_hash": "sha256:...",
  "run_id": "...",
  "raw_output": "...",
  "reasoning": "...",
  "extracted": {"line_centroid_keV": 68.7514},
  "checks": [{"id": "centroid", "pass": true, "expected": 68.75145, "got": 68.7514}],
  "item_score": 0.75,
  "item_pass": false,
  "error": null,
  "timing_s": 41.2,
  "tokens": {"prompt": 812, "completion": 640},
  "finish_reason": "stop"
}
```

`summary.json`: per-domain and per-klass counts, mean partial score with bootstrap CI
over items, pass rate with binomial CI, and a list of excluded/error rows.

### 5.6 Manifest

`build/manifest.json` records the item set name, build time, per-item hash, per-file
input hashes, the compiled `items.jsonl` hash, the taxonomy version and the schema
version. Any run that does not record the item-set hash is not comparable.

---

## 6. Generation methods

### 6.1 G1 — Extraction from verified analyses (primary)

Input: an analysis unit = a script/macro plus inputs, committed outputs and, where
available, a paper or method note. A unit is eligible only if it is completed (no
WIP-marked code, e.g. `AdditionalLevels.cpp`'s "NOT A RESULT YET" note excludes it) and
its outputs are committed.

Steps for each unit:

1. **Mechanically collect** paths, commits and input hashes. No physics interpretation.
2. **Pick a scorable decision** inside the pipeline: one fit result, one calibration
   coefficient, one classification metric, one combination, one processing step.
3. **Choose task type**:
   - `script`: hide the stage's code, give inputs and method spec, require
     `RESULTS.txt` keys; gold from the stage's committed output; executor runs the
     model's program and scores.
   - `implement`: hide one function/macro; hidden tests are copied from the repo test
     suite where one exists, otherwise the stage's output table is the oracle.
   - `compute`: give the committed fit artifacts as inputs and ask for a derived
     quantity (weighted mean, propagated uncertainty, chi2, significance, limit).
     Gold is `derived` from committed check values, computed by the builder.
   - `interpret`: give spectra and fit results, ask which component dominates an
     uncertainty or which dataset is an outlier, with the answer expressed as a key.
   - `debug`: mutate the committed script in a way the pipeline detects; the model
     repairs it; hidden tests or output comparison decides.
4. **Write the prompt** as a runner-neutral spec (section 8.2 rules).
5. **Write checks with locators**, not values.
6. **Validate mechanically** (section 7.2).
7. **Review** (section 7.3).
8. **Freeze** with provenance and state transitions.

Worked mapping for UM-ANSG v1 candidates:

- Pixel calibration (`PixelCalibration.cpp`) -> CAL coding items.
- Transfer calibration (`CalibrationLow.cpp` + `calibration_function_low_*.root`) ->
  CAL `script` items; gold from the calibration function objects and
  `ge_amxfer.result`.
- Combination and uncertainty budget (`CombineGeResult.cpp` + `ge_amxfer.result` +
  published value) -> STAT `compute` items; gold `derived` and `published`.
- Filtering/conversion (`Filter.cpp`, `ConvertBEF.cpp`) -> DAQ `implement` items.
- PSD macros and studies (`ChargeComparison.cpp`, `GateOptimization.cpp`,
  `ShapeIndicator.cpp`, Python studies) -> PSD `implement`/`interpret` items; gold from
  committed plots' underlying values where the plot is generated from a saved table,
  otherwise `recomputed` via the pinned shell.
- Half-life and Monte Carlo efficiency (78mBr) -> NUC/SIM `compute` items; gold
  `published` plus `recomputed`.
- YAG preprocessing -> DAQ `script` items.

### 6.2 G2 — Code history (SWE-style)

For a commit that changes code and tests:

- parent tree + commit subject/body as instruction (cleaned of identifiers that leak
  the answer),
- the commit's test diff as hidden tests,
- accept only if tests fail at parent and pass at commit,
- reject tests that read source text (repo rule) and tests whose mutation score is
  below the threshold (section 7.2).

This is the only method that yields hard pass/fail coding items without a data pipeline,
and it scales with the commit history of everything under `Software/`.

### 6.3 G3 — Parametric templates

For reasoning families where no committed artifact exists (e.g., kinematics, activation,
statistics):

- Write a template that randomizes parameters from a private seed.
- The template must contain **two independent solvers** (closed form and numeric, or
  two different numeric methods).
- The builder runs both, requires agreement within the check tolerance by a factor of
  at least 3, and stores both results in the item's verification record.
- The gold is the agreed value, never the generation formula's nominal parameter.
- Synthetic-generation problems follow the same rule as existing
  son-of-anton's `problems/*.toml`: known generation parameters are a hint,
  not the gold, because
  estimators are noisy. Where a fit is required, compute the expected estimator
  distribution first (or widen the tolerance from a Monte Carlo study) rather than
  guessing.

### 6.4 G4 — Teacher-model drafts

Allowed only to fill thin categories and only as drafts. Mandatory: two-path recompute,
human review, and a public/nuclear-data source for constants. A teacher draft is
rejected if no machine check can be written for it.

### 6.5 Variants

Variants multiply a single verified unit without re-authoring: shuffle run labels, hold
out one dataset, resample events, change background/source mix, perturb a parameter.
Only generate variants when a reference script can be rerun to produce new gold. Each
variant is a new item with its own hash and provenance pointing at the parent. Cap
variants per parent (for example 5) so the suite does not become one analysis repeated.

### 6.6 Anti-hallucination rules for the generator

1. Never write a numeric `expected` value. Write a `gold_ref`; the builder fills it.
2. Never paraphrase physics constants from memory; reference the artifact.
3. Never mark an item `human_verified`.
4. Never invent a file path; every path must resolve at the pinned commit or the item
   fails validation.
5. Never include answer-bearing files in `inputs`; the workspace is inspected for
   leakage (the gold values must not appear anywhere in the mounted files or prompt).
6. Flag, do not resolve, ambiguity: a `needs_human` list in the item notes.

---

## 7. Verification and curation

### 7.1 States

`draft -> machine_verified -> human_verified -> calibrated -> frozen`, with `rejected`
reachable from any state. The state lives in the item file and in the manifest. Runners
refuse to include items outside `human_verified` or later unless `--allow-draft` is
passed for harness debugging.

### 7.2 Machine gates (builder)

| Gate | Reject condition |
|---|---|
| Schema | Any missing required field or unknown type. |
| Locator resolution | File absent at pinned commit, locator matches zero or multiple rows, extracted value != stored expected. |
| Input hashes | Any input hash mismatch against the manifest. |
| Leakage | Any expected value appears in the prompt or mounted inputs (normalized numeric search). |
| Consistency | Every `output.keys` entry has at least one check; every check key is in `output.keys`; prompt text and check units agree; prompt states the required key names. |
| Tolerance sanity | `tol.abs > 0.1 * abs(expected)` and no note; `rel` and `abs` both null; tolerance wider than the gap between adjacent plausible answers. |
| Duplicate | Normalized prompt n-gram overlap above threshold or embedding cosine above threshold against the same set. |
| Code tests | Hidden tests fail at parent, pass at commit; mutation score below 0.6; test reads source text. |
| Sandbox | Reference solution fails to run in the declared environment within the timeout. |

The yap-run check set is the negative example to test this gate against: the prompt
asked for ADC-channel peak positions while the checks held keV energies, and only 6 of
13 requested keys were scored. Both conditions must reject an item.

### 7.3 Human review (per item)

- Can the gold be reproduced from the referenced artifact? (spot-check the builder's
  extraction, not the model's summary.)
- Is every convention needed to get within tolerance stated? Units, fit window,
  background model, weighting, calibration order, empty-bin handling.
- Does the task correspond to work actually done, not a textbook exercise?
- Are inputs free of the solution and of other items' answers?
- Is `sensitivity` correct, and is the prompt free of restricted detail?
- Given a pilot run, is it discriminating (section 11) or trivial/impossible?

Review is recorded as `{reviewer, date, method, notes}`. The generating model and the
reviewing model must not be the same run; for v1, the human is the final reviewer of
every check's gold reference and every prompt.

### 7.4 Sandbox and reference environments

- Runner A executes code items in `bwrap`: no network, read-only inputs, writable
  workspace, wall-clock limit per item, memory cap.
- Analysis environments come from the pinned UM-ANSG (or other) flake via
  `nix develop .#<shell> -c ...`; the resolved `flake.lock` hash is recorded in the run.
- The reference solution (for builder validation) and the model solution run in the same
  environment. Environment drift invalidates a run, not a result.
- Heavy pipelines run on a tiered budget: `fast` <= 120 s, `standard` <= 15 min,
  `full` longer and only for `pipeline` horizon items.

---

## 8. Runner A specification

### 8.1 Commands

```
runner_a run    --items build/items.jsonl --endpoint URL --model NAME \
                --api chat|completion --profile NAME --seeds 1,2,3 \
                --concurrency 8 --limit N --out results/<run>
runner_a exec   --run results/<run>            # sandbox execution for code items
scores/scorer.py --run results/<run> --items build/items.jsonl
scores/compare.py --runs results/<a> results/<b> --out compare/
```

`run` and `exec` are separate so scoring can be iterated without re-querying models.
The HTTP layer, retry logic, concurrency and JSONL streaming are lifted from
`gsm8k_eval.py`; the GSM8K task stays available as its own item set built the same way,
which preserves the AMD calibration comparison.

### 8.2 Prompt assembly

- Each item carries its complete prompt. Runner A wraps it only with the output-format
  epilogue needed for the chosen contract (`results_txt_v1`: "End with one
  `key = value` line per required key").
- `api: chat` sends system + user; `api: completion` sends the raw prompt with the
  item's stop sequences. A comparison run must use one API mode for all models and
  record it. The default for v1 is `chat`, because the target checkpoints are
  chat/reasoning tuned; GSM8K keeps its own completion configuration.
- Sampler profiles are runner-level, not item-level: `greedy` (temperature 0),
  `thinking`, `nothink`. The primary model comparison uses one profile, with repeats at
  a second seed. Item-level overrides are allowed only for `max_tokens` and only when
  every compared model gets the same override.

### 8.3 Executors

| Executor | Use | Behavior |
|---|---|---|
| `none` | compute/interpret | Parse the raw answer into keys. |
| `python_sandbox` | script | Extract the last runnable code block, execute with inputs mounted, collect `RESULTS.txt`, score. |
| `repo_sandbox` | implement/debug | Materialize the model's files/patch in a checkout of the pinned repo, run hidden tests. |
| `files` | pipeline | Collect declared artifacts, then apply checks. |

Execution failures are scored as failures, not errors, if the model produced runnable
output; infrastructure failures are recorded with `error` and excluded from the summary
(and reported).

### 8.4 Output-format parsing

`results_txt_v1` accepts lines of the form `key = value`, one per line, with arbitrary
surrounding prose; parsing is case-insensitive on keys, tolerant of whitespace and
scientific notation, and rejects ambiguous duplicate keys. Extraction tests are part of
the harness test suite; a parser change triggers a full rescore of stored raw outputs.

---

## 9. Runner B specification (Autophysicist, later)

Prerequisite: `son-of-anton/STATUS.md` item 9 resolved and the qualification suite
passing.

Adapter contract:

1. Convert an item into a problem spec: workspace snapshot, prompt, declared runtime
   and token budget, required output contract.
2. Pin the `son-of-anton` commit, the model route, the sandbox configuration and the
   event/token caps in `run.json`.
3. Run with the same item set hash; normalize its outputs into the `items.jsonl`
   schema; apply the same scorer.
4. Qualification before any comparison: the three shipped toy problems
   (`cobalt_calibration`, `bromine_halflife`, `yap_psd`) must pass all checks with the
   reference model; a failing qualification invalidates Runner B results rather than
   the models.
5. Runner B runs never establish gold, and a Runner B score is only compared against
   Runner A on identical items and identical item-set hash.

---

## 10. Metrics and comparison

- Primary metric: mean partial credit over required checks per item, averaged over
  items, reported per domain and per `klass`.
- Secondary: item pass rate (`all required checks pass`), exact binomial CI.
- Paired comparison (the one that matters): for two models on the same items, report
  the mean per-item score difference with a bootstrap CI over items, and McNemar's
  exact test on item pass/fail. Unpaired CIs are misleading at this N.
- Repeats: k seeds at fixed sampler; report per-item variance and use paired seeds
  across models. A ranking that flips within seed noise is reported as a tie.
- Runner comparison: same paired machinery on `score_B - score_A`, per domain, plus a
  check that item-level rankings are consistent (Spearman) between runners.
- Never collapse to one headline number without the domain table.

---

## 11. Calibration

1. Pilot with the strongest available checkpoint and a deliberately weak one; score the
   set. Items everyone always passes or always fails are candidates for retirement
   (kept in a reserve pool, not deleted, because new checkpoints may find them hard).
2. Target discrimination: per item, pass probability between roughly 0.2 and 0.9 for
   the reference checkpoint; category averages should not sit at 0 or 1.
3. Test-retest: same model and sampler, different seed. If per-item flips exceed the
   predicted binomial noise, the item's parsing or tolerance is broken.
4. Difficulty metadata (`difficulty`) is overwritten from pilot statistics; the
   initial estimate is only a hint.
5. Freeze `knoll-fast-v1` and `knoll-full-v1`; open a new version rather than editing a
   frozen set, and re-run the old set when models change to keep longitudinal numbers
   comparable.

---

## 12. Privacy and contamination

- The private lab data and restricted analyses are the main contamination defense:
  no public model has seen them. Keep prompts and items local unless a public variant
  is deliberately created.
- The `test/` split is not used for prompt iteration. Only `dev/` is.
- Before any item is marked `publishable`, run a similarity search against public
  benchmarks and a web search of a distinctive phrase; drop or rewrite on a match.
- Sanitize absolute paths, hostnames and unreleased numbers from publishable items and
  from `run.json` before sharing results.
- Do not commit `items-local/`, `sources.env`, `results/`, or anything derived from
  MUSIC without an explicit sensitivity review.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wrong gold injects noise | Locator-based gold, builder re-extraction, human spot-check, two-path recompute for derived values. |
| Scaffold coupling | Item/runner split, Runner B qualification, never use runs as gold. |
| Ambiguous prompts | Consistency gate plus "reject, don't loosen tolerance". |
| Contamination | Private data, local-only test split, similarity checks before publish. |
| Overfitting to dev | Held-out test split, frozen sets, new version on change. |
| Weak hidden tests | Mutation score threshold and fail-at-parent/pass-at-commit. |
| Parser brittleness | Extraction unit tests, stored raw outputs for rescoring. |
| Data drift | Input hashes in the manifest; mismatch invalidates. |
| Heavy pipelines | Tiered runtime budgets, fast suite as the primary comparison. |
| Runner B still buggy | Runner A is primary; Runner B gated on qualification. |

---

## 14. Roadmap

Phase 0 — Inventory and schema.
Local model produces `inventory/um-ansg.inventory.toml` and a local MUSIC inventory
using the Appendix B contract. Finalize `schema/*.json` and the item/check examples.

Phase 1 — Builder and Runner A plus a pilot.
Implement `build.py`, `validate.py`, `runner_a.py`, `scorer.py`, `compare.py`, the
sandbox, and 20-30 human-reviewed `fast` items covering CAL, PSD, STAT, SW, and one
real-data `compute` item. Pilot on two checkpoints.

Phase 2 — Scale.
Grow to 100-150 `fast` items plus 30-50 `standard`/`full` items via G1-G3; retire
non-discriminating items; freeze `knoll-fast-v1` and `knoll-full-v1`.

Phase 3 — Runner B.
Once STATUS item 9 is resolved, add the adapter and pass qualification; run the
model-by-runner matrix and report per-domain deltas.

Phase 4 — Operations.
Re-run frozen sets for each new quant/checkpoint, keep the manifest and raw outputs,
refresh contamination checks, and open v2 only when the item set must change.

---

## 15. Open questions

1. Repo location (`knoll-bench` is the working name) and whether it is a separate repo
   that reuses `gsm8k-eval`'s HTTP layer or a second command inside `gsm8k-eval`.
2. Primary API mode for v1 (`chat` proposed) and max-token budget per horizon.
3. Whether `RESULTS.txt` stays the canonical contract or a JSON sidecar is added for
   machine parsing (the human contract should stay simple).
4. How many restricted items may ship in the local overlay versus being reformulated on
   public data.
5. Whether hidden tests for `implement` items come from repo test files directly or are
   rewritten by the generator and then mutation-tested.

---

## Appendix A — Worked examples

The numeric gold fields below are placeholders resolved at build time, by design. The
builder replaces `"<resolved at build>"` with the value extracted through `gold_ref`.

### A.1 Real-data reasoning/compute item (abridged)

See section 5.2. Provenance chain for the centroid check:
`UM-ANSG@9fc4510:73mGe/results/ge_amxfer.result`, row
`Cu_Shield_Signal_10%_(01/13)`, column `ge_mu`; published counterpart in the 73mGe
paper. The prompt is generated from the method spec, never copied from the result file,
and the result file is not mounted.

### A.2 Coding item from Analysis-Utilities (abridged)

```toml
id = "analysisutilities-init-plots-dir-trailing-slash"
klass = "coding"
domain = "SW"
type = "implement"
horizon = "single_turn"

[source]
repo = "Analysis-Utilities"
commit = "<pinned>"
paths = ["python/analysis_utilities/tests/test_init_utils.py", "doc/pages/python.md"]

prompt = '''
Implement the plotting-directory configuration used by analysis_utilities.
The function set_plots_base_dir must store the base directory with any trailing
slash removed, and GetPlotsBaseDir must return exactly that string. Existing
behavior and signatures are described in the mounted notes; do not change other
semantics. Deliver the patch in the workspace.
'''

[env]
nix_develop = "Analysis-Utilities"
network = false
timeout_s = 600

[tests]
kind = "pytest"
command = "python -m pytest python/analysis_utilities/tests/test_init_utils.py -k trailing_slash"
hidden_files = ["python/analysis_utilities/tests/test_init_utils.py"]
expected_fail_at_parent = true
expected_pass_at_commit = true

[[checks]]
id = "hidden-test"
key = "pytest_exit"
type = "integer"
expected = 0
required = true

[checks.gold_ref]
kind = "manual"
note = "test outcome, not a physics value"

[verification]
state = "draft"
machine = []

[verification.human]
```

### A.3 Pipeline item (abridged)

From YAP-PSD: implement a documented tail-to-total or charge-comparison feature and
apply it to a mounted waveform sample; gold is the committed per-population statistic
(`recomputed` via the pinned `.#YAP-PSD` shell), checked with a tolerance justified by
the statistical uncertainty of the sample. `horizon: tool_loop`, executor `files`.

---

## Appendix B — Generator contract (prompts for the local model)

The local model runs these as separate passes. They are provided here so the spec is
self-contained; the prompts themselves live in `generators/*.md` once implemented.

**Pass 1, inventory.** "You are indexing a completed analysis repository. For each
analysis unit (one macro/script/stage) emit: path, language, inputs (paths, ROOT trees,
hashes if computable), outputs (paths and whether committed), the command that produced
them if present, the environment (flake shell), whether the stage is marked work in
progress, and a one-line factual description of what it computes. Quote file contents
for evidence. Do not interpret physics beyond what the files state. Emit the inventory
schema exactly. List units you cannot classify under `needs_human`."

**Pass 2, extraction.** "Select one scorable decision from this unit. Produce one item
TOML matching the schema. Rules: (1) never write a numeric expected value, write a
`gold_ref` locator instead; (2) every mounted input path must exist at the pinned
commit; (3) the prompt must state units, fit/selection conventions and the exact output
keys; (4) the prompt must not contain any answer value and must not mount answer-bearing
files; (5) if two competent analyses could disagree beyond the tolerance, mark the item
`needs_human` and stop. Provide the provenance fields."

**Pass 3, self-review.** "Re-read the item and the referenced artifact. Check the
prompt/check consistency gate, the leakage gate and the tolerance sanity gate from the
plan. Report failures as a list; do not edit the item." A different model or the human
performs the real review.

---

## Appendix C — Human review checklist

- [ ] Every `gold_ref` resolves to a committed artifact or publication I trust.
- [ ] The expected value was extracted, not typed.
- [ ] The prompt states every convention needed to hit the tolerance.
- [ ] Units and key names are unambiguous and match `output.keys`.
- [ ] No answer value or answer-bearing file is mounted or quoted.
- [ ] The task is one I actually perform, at a difficulty worth measuring.
- [ ] `sensitivity` and `publishable` are correct.
- [ ] Hidden tests fail at parent and pass at commit; mutation score acceptable.
- [ ] The item is tagged with a plausible domain, type, horizon and runtime.
- [ ] If derived or manual gold, an independent path or note justifies it.
