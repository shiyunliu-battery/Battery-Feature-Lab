# Evidence Layer Roadmap for BFL

My assessment is:

> **The current BFL has already completed the foundation from battery time-series data to feature tables, diagnostic summaries, and LLM context.**
> However, it has not yet fully implemented the paper-level idea of **protocol-aware evidence selection and claim-level verification**.
> Therefore, the project does not need to be overturned or rebuilt. It needs an additional **Evidence Layer** on top of the existing BFL framework.

In other words:

```text
Current BFL:
raw battery data -> normalized data -> feature tables -> diagnostic tags -> llm_context.jsonl

Complete framework needed for the paper:
raw battery data -> protocol-aware segmentation -> feature/evidence candidates
-> minimal evidence selection -> LLM explanation -> claim-level verification
```

## 1. What Has Current BFL Already Completed?

### Completed 1: Standardized Input for Raw Battery Data

BFL already supports CSV, TSV, JSON, JSONL, Parquet, and similar inputs. It normalizes columns from different data sources into standard fields such as:

```text
time_s, voltage_v, current_a, temperature_c,
charge_capacity_ah, discharge_capacity_ah,
cycle_index, step_index, step_type
```

The README explains that BFL can infer some missing information from the file name, current sign, and command-line parameters.

This means BFL has already completed the first layer:

> **Turning messy battery cycling data into processable normalized time series.**

This foundation matters because without normalization, downstream feature extraction and evidence selection cannot work reliably.

### Completed 2: Battery Feature Extraction Backbone

BFL already has a relatively complete feature extraction system, including:

- cycle summaries;
- early-life Delta-Q features;
- ICA/DVA features;
- relaxation features;
- stress features;
- EIS descriptors;
- rule-based degradation tags;
- JSONL context summaries.

These are clearly listed in the README.

From the code perspective, `FeaturePipeline` already connects these featurizers:

```text
CycleSummaryFeaturizer
DeltaQFeaturizer
ICADVAFeaturizer
RelaxationFeaturizer
StressHistogramFeaturizer
EISDRTFeaturizer
```

It then generates degradation tags and LLM context.

Therefore, BFL has already completed:

> **raw time series -> battery feature tables**

This can serve as the **feature backbone** for the research direction in the paper.

### Completed 3: Initial LLM Context Generation

BFL no longer only outputs Parquet feature tables. It also outputs:

```text
llm_context.jsonl
```

The README says this file includes dataset overview, data-quality warnings, capacity and efficiency trends, selected Delta-Q/ICA-DVA/relaxation/stress highlights, diagnostic evidence, analysis configuration, and reliability notes.

The code also shows that `llm_json_writer.py` generates fields such as:

```text
summary
dataset_overview
data_quality
cycle_life_summary
feature_highlights
diagnostic_evidence
features
review_notes
provenance
```

This shows that current BFL already has:

> **feature-to-context capability.**

This is important because LLM grounding does not need to start from zero. BFL can already compress battery data into structured summaries readable by an LLM.

### Completed 4: Data Quality and Reliability Notes

This is also important. BFL does not simply calculate features; it has already started to consider whether those features are reliable.

For example, `llm_json_writer.py` checks:

- usable cycle count;
- partial or unbalanced cycles;
- capacity reliability;
- whether the reference cycle exists;
- whether the target cycle exists;
- feature computability.

If no usable complete cycles exist, it warns that capacity and trend interpretation are not reliable.

This shows that BFL already contains an early version of the idea:

> **Not all features should be trusted equally.**

That idea is close to the evidence selection layer that needs to be added next.

## 2. What Has Current BFL Not Completed Yet?

Although BFL is already close, it has not yet completed the core innovation required by the paper.

### Gap 1: No True Protocol-Aware Segmentation Yet

Current BFL uses `step_type`, and it can infer charge/discharge/rest from current sign and rest thresholds. The README also states that the input should contain at least time, voltage, current, and enough cycle/step information to identify charge, discharge, and rest periods.

However, this is not yet the protocol-aware segmentation described in the paper.

Current BFL is closer to:

```text
charge / discharge / rest / step type
```

The paper needs:

```text
CC-CV
HPPC
GITT
pulse
rest
OCV relaxation
constant-current discharge
constant-voltage tail
pulse recovery
mixed protocol segment
```

These are different levels of semantics.

Charge/rest/discharge are low-level step labels. HPPC/GITT/CC-CV are experimental protocol semantics.

This layer needs to be added.

### Gap 2: `llm_context.jsonl` Is a Compact Summary, Not a Selected Evidence Pack

The current `llm_context.jsonl` output is valuable, but it is closer to:

> Summarizing existing features for the LLM.

It is not yet:

> Selecting a minimal, sufficient, non-redundant, protocol-consistent evidence set based on a user question.

In other words, current BFL performs **feature/context generation**, not **question-aware evidence selection**.

The difference is:

| Layer | Current BFL | Needed for the paper |
| --- | --- | --- |
| Feature extraction | Complete | Keep |
| Diagnostic labels | Rule-based tags exist | Use as evidence candidates |
| LLM context | Compact summary exists | Needs question-aware selection |
| Token cost control | Not yet a core objective | Needs to become an optimization objective |
| Redundancy penalty | Not explicit yet | Needs to be added |
| Evidence sufficiency | No formal evaluation yet | Needs to be added |

Therefore, BFL has not yet completed evidence selection. It has completed the layer before evidence selection.

### Gap 3: No Claim-Level Verifier Yet

Current BFL has reliability notes and data-quality warnings, which is good. But it does not yet perform:

```text
LLM outputs one sentence
        ->
split it into claims
        ->
classify each claim as measured fact / derived feature / supported hypothesis / unsupported claim
        ->
check whether the claim is supported by evidence or literature
```

This is the most important part of **claim-level faithfulness verification** in the paper.

Current BFL can tell you:

> Whether the data quality is reliable, which features exist, and which diagnostic evidence signals are present.

But it cannot yet tell you:

> Whether a specific mechanistic explanation is over-inferred.

For example, if the LLM says:

> The degradation is caused by lithium plating.

Current BFL may have evidence such as capacity fade, relaxation change, and ICA peak shift, but it does not yet have a verifier that decides:

```text
Is this claim supported by the current evidence?
Is it only a plausible hypothesis?
Should it be downgraded to "may indicate"?
Should it be marked as an unsupported mechanism claim?
```

Therefore, the claim verifier needs to be newly added. It cannot be solved by simply editing an existing file.

## 3. Does the Framework Need to Be Rewritten or Extended?

My recommendation is:

> **Do not heavily rewrite the lower-level BFL framework. Add an Evidence Layer on top of the existing framework.**

In other words, do not overturn the current structure.

The current structure is already reasonable:

```text
reader
  ->
normalized_timeseries
  ->
featurizers
  ->
feature tables
  ->
degradation tags
  ->
llm_context.jsonl
```

The missing layer is:

```text
feature tables + normalized_timeseries + degradation tags + user question
        ->
evidence candidates
        ->
evidence scoring
        ->
minimal evidence selection
        ->
evidence pack
        ->
LLM answer
        ->
claim verification
```

More precisely:

> **This is not a modification of BFL's main framework. It is an extension of BFL's research framework.**

## 4. Five Directions to Add

### Direction 1: Protocol-Aware Segmentation

Add a new module:

```text
battery_feature_lab/protocol/
  protocol_detector.py
  segment_classifier.py
  protocol_schema.py
```

It should identify:

- CC;
- CV;
- CC-CV;
- HPPC pulse;
- pulse recovery;
- GITT pulse;
- GITT relaxation;
- rest;
- discharge;
- abnormal segment.

Why add this?

Current BFL has many features, but the meaning of a feature depends on the protocol. For example, a relaxation slope in GITT may relate to diffusion or OCV relaxation; in HPPC it may be closer to pulse recovery; in ordinary rest it may only represent static recovery.

Without protocol-aware segmentation, the LLM can easily interpret features incorrectly.

### Direction 2: Evidence Candidate Generation

Add a new module:

```text
battery_feature_lab/evidence/candidates.py
```

This module should convert existing feature tables into evidence objects.

Example:

```json
{
  "evidence_id": "relax_cycle_80_slope",
  "type": "derived_feature",
  "source_table": "relaxation_features",
  "cycle_index": 80,
  "protocol": "rest_after_discharge",
  "feature_name": "relaxation_slope",
  "value": 0.0032,
  "unit": "V/h",
  "interpretation_hint": "increased relaxation slope may indicate stronger polarization",
  "reliability": "medium",
  "token_cost": 42
}
```

Why add this?

Current BFL has feature tables and LLM context, but it does not yet have a unified evidence representation. If the paper claims "evidence selection", it must first define what evidence is.

This step upgrades:

```text
feature
```

into:

```text
evidence
```

They are not the same. A feature is a numerical value. Evidence is a factual unit with context, source, protocol, reliability, and interpretation boundaries.

### Direction 3: Question-Aware Evidence Scoring

Add a new module:

```text
battery_feature_lab/evidence/scorer.py
```

The scoring logic can start as:

```text
score = relevance_to_question
      + protocol_consistency
      + local_context_importance
      + mechanism_support
      + reliability
      - redundancy
      - token_cost
```

Why add this?

Current BFL outputs a general summary. It gives a similar context package regardless of the user's question.

The paper direction requires:

> Different questions should select different evidence.

For example, if the user asks:

> Why did capacity fade after cycle 80?

The system should prioritize:

- capacity trend;
- ICA/DVA peak shift;
- relaxation change;
- resistance-related tags;
- high-SOC stress exposure;
- relevant cycle windows around cycle 80.

But if the user asks:

> Is this dataset suitable for SOH modelling?

The system should prioritize:

- cycle completeness;
- capacity reliability;
- number of usable cycles;
- missing columns;
- reference/target cycle suitability;
- temperature/current variability.

This is question-aware selection.

### Direction 4: Minimal Evidence Set Selection

Add a new module:

```text
battery_feature_lab/evidence/selector.py
```

The first version does not need to be complex. It can use greedy selection:

```text
1. Sort by score.
2. Select the most valuable evidence item each time.
3. Skip evidence that is too redundant with already selected evidence.
4. Stop when required evidence types are covered or the token budget is exhausted.
```

Later, this can be upgraded to:

- submodular optimization;
- integer programming;
- learning-to-rank.

Why add this?

The core of the paper is not "we have many features". It is:

> **We can use less evidence to achieve equal or higher explanation quality.**

That requires a selection mechanism.

Otherwise, reviewers may say:

> You are simply summarizing all features and giving them to the LLM.

### Direction 5: Claim-Level Verifier

Add a new module:

```text
battery_feature_lab/verification/
  claim_parser.py
  claim_classifier.py
  evidence_matcher.py
  verification_report.py
```

The output can look like:

```json
{
  "claim": "The capacity loss is likely related to resistance growth.",
  "claim_type": "supported_hypothesis",
  "supporting_evidence": [
    "hppc_ir_drop_increase",
    "relaxation_slope_increase",
    "degradation_tag_resistance_growth"
  ],
  "support_level": "medium",
  "warning": "No direct EIS evidence available."
}
```

Why add this?

The most dangerous part of battery interpretation is mechanistic attribution. LLMs can easily turn "possibly related to" into "caused by".

The verifier's value is to constrain the LLM:

```text
If it can only state a measured fact, do not let it claim a mechanism.
If it can only state a supported hypothesis, do not let it claim a confirmed diagnosis.
If there is no evidence, the claim must be marked unsupported.
```

This is also the key difference between the paper and ordinary RAG.

## 5. What Stage Is BFL Currently At for the Paper?

I would position it as follows:

| Module | Current BFL status | Enough for the paper's main line? |
| --- | --- | --- |
| Data reading and normalization | Exists | Enough |
| Basic feature extraction | Exists | Enough |
| Delta-Q / ICA / DVA / relaxation / stress features | Exists | Enough as backbone |
| Degradation tags | Exists | Can serve as evidence candidates |
| LLM context summary | Exists | Has an initial form, but not enough |
| Protocol-aware segmentation | Not enough | Needs to be added |
| Evidence object schema | Not enough | Needs to be added |
| Question-aware scoring | Missing | Needs to be added |
| Minimal evidence selection | Missing | Needs to be added |
| Claim-level verifier | Missing | Needs to be added |
| Evaluation baselines | Needs systematization | Needs to be added |

Therefore, I would not say BFL is simply "unfinished". More precisely:

> **BFL has completed the feature extraction and context generation backbone. The paper needs to add evidence representation, protocol awareness, evidence selection, and claim verification on top of it.**

## 6. Why This Is Not a Rewrite

The current BFL pipeline is already correct. It has a complete path from input data to feature tables and then to LLM context.

Rewriting it would waste the work already done.

The real issue is not that the lower-level pipeline is wrong. The issue is that it lacks one layer:

```text
features -> evidence -> selected evidence -> verified explanation
```

Therefore, the most reasonable change is:

```text
keep the original BFL pipeline
        +
add an evidence layer
        +
add a verification layer
```

Instead of:

```text
build a new Battery-RAG repository from scratch
```

## 7. How to Adjust the README Positioning

The current first sentence of the README is:

> BFL extracts battery cycling features from BDS-style exports and common cycler tables. It turns raw time-series data into feature tables for SOH/RUL modeling, feature screening, explainability, and compact diagnostic summaries.

This sentence is already good. It does not need to be removed.

But a research extension sentence can be added:

```text
BFL also provides the feature backbone for evidence-grounded LLM interpretation of battery time-series data. In this extension, extracted features are converted into protocol-aware evidence candidates, selected under token and redundancy constraints, and used for faithful claim-level battery diagnostics.
```

This upgrades BFL's positioning from:

```text
feature extraction tool
```

to:

```text
feature extraction + evidence-grounded interpretation backbone
```

without making it look like the project suddenly became a chatbot project.

## 8. Final Recommendation

### Do Not Heavily Rewrite the Framework

Current BFL has already completed:

```text
raw data normalization
feature extraction
diagnostic tagging
compact LLM context generation
data quality / reliability notes
```

These are a strong backbone.

### Add Five Directions

The next work should add:

```text
1. protocol-aware segmentation
2. evidence candidate schema
3. question-aware evidence scoring
4. minimal evidence set selection
5. claim-level verification
```

### Recommended Paper Framing

Do not say:

> We developed an LLM-based battery analysis tool.

Instead, say:

> We developed a protocol-aware evidence selection framework built on Battery Feature Lab, which compresses battery time-series data into minimal, task-relevant, and verifiable evidence for faithful LLM-based battery diagnostics.

This is the most stable framing. BFL does not need to be overturned. It needs to evolve from a **feature lab** into an **evidence-grounded battery interpretation framework**.

## 9. Phase 1 Implementation Scope

The first implementation phase should be deliberately narrow:

```text
evidence schema
-> evidence candidate generation
-> deterministic question-aware scoring
-> greedy evidence selection
-> selected evidence output
```

This phase should not attempt to solve protocol-aware segmentation or full claim verification yet. Those require more domain assumptions and evaluation data.

### Phase 1 Goals

Phase 1 adds a local Evidence Layer that converts existing BFL outputs into structured evidence records.

Inputs:

```text
normalized_timeseries
cycle_features
delta_q_features
ica_dva_features
relaxation_features
stress_features
degradation_tags
optional user question
token budget
```

Outputs:

```text
evidence_candidates.parquet
selected_evidence.parquet
evidence_candidates.jsonl
selected_evidence.jsonl
```

The Parquet outputs support downstream data analysis. The JSONL outputs are convenient for LLM/RAG-style use.

### Phase 1 Evidence Schema

Each evidence record should carry enough provenance and interpretation context to be reviewed without reopening the raw feature table.

Core fields:

| Field | Meaning |
| --- | --- |
| `evidence_id` | Stable identifier for the evidence item |
| `evidence_type` | `measured_fact`, `derived_feature`, or `diagnostic_tag` |
| `source_table` | Original BFL table, such as `cycle_features` or `degradation_tags` |
| `cycle_index` / `cycle_start` / `cycle_end` | Cycle scope, when applicable |
| `feature_name` | Source feature or diagnostic signal |
| `value` and `unit` | Numeric value and unit, when applicable |
| `protocol` | Protocol label; `unknown` in Phase 1 |
| `support_role` | Role such as `capacity_trend`, `ica_peak_shift`, or `usage_stress` |
| `reliability` | Conservative reliability label |
| `interpretation_hint` | What this item can and cannot support |
| `text` | Compact natural-language evidence statement |
| `token_cost` | Approximate token cost for selection |
| `score` | Question-aware score |
| `selection_rank` | Rank inside the selected evidence pack |

### Phase 1 Candidate Generation

Candidate generation should be conservative and table-driven:

- `cycle_features`: capacity retention, efficiency, CV fraction, and cycle-level capacity facts.
- `delta_q_features`: Delta-Q area, variance, and norm features between reference and target cycles.
- `ica_dva_features`: first-to-last ICA/DVA peak and area changes.
- `relaxation_features`: first-to-last relaxation tau, voltage recovery, and slope changes.
- `stress_features`: high-SOC rest, C-rate, throughput, and temperature context.
- `degradation_tags`: existing rule-based diagnostic tags as evidence candidates, not final diagnoses.

The important distinction is:

```text
feature value -> evidence object with source, context, reliability, and interpretation boundary
```

### Phase 1 Scoring

Scoring can start as a deterministic heuristic:

```text
score =
    question relevance
  + source prior
  + reliability
  + protocol consistency
  - token cost penalty
```

Question relevance can be keyword/role based in Phase 1. For example:

- capacity/fade/SOH questions prioritize capacity trend, Delta-Q, ICA/DVA, and diagnostic tags.
- mechanism/why questions prioritize diagnostic tags, ICA/DVA, relaxation, stress, and capacity trend.
- dataset suitability questions prioritize cycle completeness, capacity reliability, stress context, and data-quality-adjacent evidence.

This is not a learned ranker. It is a transparent baseline that can later be replaced by learning-to-rank or submodular optimization.

### Phase 1 Selection

Selection should use greedy ranking:

```text
1. Sort candidates by score.
2. Add the highest-scoring candidate if it fits the token budget.
3. Skip candidates with the same redundancy key.
4. Limit over-selection from one source table.
5. Stop at max selected items or token budget.
```

This is enough to prove the basic mechanism:

> BFL can produce a compact selected evidence pack instead of passing every feature summary to the LLM.

### Phase 1 Non-Goals

Phase 1 should not claim to complete:

- HPPC/GITT/CC-CV protocol recognition;
- protocol-aware evidence semantics;
- claim parsing from arbitrary LLM answers;
- claim-level support classification;
- literature-grounded mechanism verification;
- paper-level evaluation baselines.

Those belong to later phases.

### Phase 1 Expected Difference From Current BFL

Current BFL produces:

```text
feature tables + degradation tags + llm_context.jsonl
```

After Phase 1, BFL additionally produces:

```text
evidence_candidates
selected_evidence
```

The conceptual shift is:

```text
summarize all relevant features
```

to:

```text
construct evidence objects and select a compact question-aware evidence pack
```

This is the smallest practical step toward the full paper framework.
