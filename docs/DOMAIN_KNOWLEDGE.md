# Battery Feature Domain Knowledge for LLM Analysis

This document defines how an LLM should interpret extracted battery features from
Battery Feature Lab. It is intended to be used as retrieval/context material for
analysis of lithium-ion battery raw-data features, degradation evidence, SOH/RUL
signals, and experimental or field-use stress factors.

The LLM must treat feature-derived interpretations as evidence-weighted
hypotheses, not as direct proof of electrochemical mechanisms unless the input
contains mechanism-specific diagnostics such as validated OCV fitting, reference
electrode measurements, post-mortem analysis, or chemistry-specific validated
ICA/DVA maps. [R1], [R6], [R7]

## Core Reasoning Policy

- Always separate measured feature facts from inferred degradation explanations.
  Example: "discharge capacity decreased with significant negative Sen's slope"
  is a feature fact; "loss of lithium inventory contributed" is an inferred
  hypothesis unless independently validated. [R1], [R6], [R11], [R12]

- Do not infer a unique degradation mode from capacity fade alone. Capacity fade
  is a cell-level consequence of multiple possible physical and chemical
  mechanisms, often summarized into loss of lithium inventory (LLI), loss of
  positive-electrode active material (LAM_PE), and loss of negative-electrode
  active material (LAM_NE). [R6], [R12]

- Prefer trend significance over single-threshold claims. When time/cycle
  sequences are available, use non-parametric trend evidence such as
  Mann-Kendall p-value and Sen's slope, and state the p-value, slope, and sample
  count. [R13], [R14]

- Treat SHAP, feature importance, correlation, and mutual information as model
  explanation tools, not causal proof. They can identify features associated
  with predictions, but chemistry/protocol context and validation are required
  before making mechanism claims. [R3], [R8], [R10]

- Explicitly report when a conclusion is chemistry-, protocol-, temperature-, or
  dataset-dependent. Battery lifetime prediction features can encode cycling
  protocol information, and this can create information leakage for some use
  cases such as production quality control on identically cycled cells. [R3]

- Do not compare absolute current variance across cells with different nominal
  capacities. Normalize current to C-rate before comparing dynamic-current
  stress or variance. [R9], [R10]

- If datasheet limits are not provided, do not label a C-rate as unsafe or
  excessive in absolute terms. State only the observed C-rate distribution or
  its position relative to the batch distribution. [R9], [R10]

- For LLM summaries, use compact evidence statements instead of dumping all
  feature columns. Mention the most relevant feature names, values, confidence,
  and missing diagnostics. [R2], [R3], [R8]

## Feature Interpretation Rules

### 1. Capacity, Energy, Efficiency, and EFC Features

- `discharge_capacity_ah` and normalized discharge capacity are direct
  indicators of available cell capacity under the measurement protocol; use them
  to describe SOH trajectory only under comparable test conditions. [R2], [R6]

- A statistically significant negative trend in normalized discharge capacity is
  valid evidence of capacity fade, but not enough by itself to assign LLI,
  LAM_PE, or LAM_NE uniquely. [R6], [R12], [R13], [R14]

- `coulombic_efficiency` can indicate charge balance and side-reaction-related
  loss when measured accurately, but small deviations can also come from
  measurement resolution, integration error, protocol differences, or data
  cleaning. Use CE as supporting evidence, not as standalone diagnosis. [R6],
  [R12]

- `energy_efficiency`, voltage hysteresis, and increasing apparent resistance
  are consistent with power/impedance degradation, but the LLM should avoid
  identifying a specific resistance source unless EIS, pulse, or validated model
  features support it. [R6], [R11]

- `equivalent_full_cycles` (EFC) is useful for normalizing usage throughput
  across partial cycles, but EFC alone does not capture calendar aging,
  temperature exposure, SOC window, rest periods, or current dynamics. [R9],
  [R10]

### 2. Early-Life Delta-Q(V) Features

- `delta_q_features` compare two early discharge capacity-voltage curves, e.g.
  `Q_target(V) - Q_reference(V)`. In Severson et al., early-cycle discharge
  voltage-curve features predicted cycle life before substantial capacity fade
  appeared in LFP/graphite cells cycled under fast-charging protocols. [R2]

- `delta_q_variance`, `delta_q_log_variance`, `delta_q_min`, `delta_q_area`,
  and voltage-window statistics should be interpreted as early curve-shape
  descriptors associated with lifetime prediction, not as direct measurements of
  one degradation mechanism. [R2], [R3]

- If the reference and target cycles are not both present, or if curve overlap is
  small, the LLM should mark Delta-Q evidence as unavailable or weak rather than
  extrapolating. [R2]

- Delta-Q features are most defensible when cycles are measured under comparable
  discharge current, temperature, voltage limits, and rest conditions. Protocol
  differences can change voltage curves independently of degradation state.
  [R2], [R3]

- When using Delta-Q features for prediction across protocols, distinguish
  whether the task is protocol selection, early lifetime forecasting, or
  cell-to-cell quality variation. Features that encode protocol may be useful in
  protocol optimization but inappropriate for protocol-blinded quality control.
  [R3], [R4]

### 3. ICA and DVA Features

- Incremental capacity analysis (ICA, `dQ/dV`) and differential voltage analysis
  (DVA, `dV/dQ`) are voltage-curve derivative methods used for non-invasive
  battery diagnosis and prognosis, including state-of-health estimation, because
  curve variations relate to electrode thermodynamics and chemistry. [R7], [R15]

- ICA/DVA peak position, height, width, area, and prominence can provide
  evidence of changing electrochemical signatures, but their interpretation is
  chemistry-specific and sensitive to current rate, temperature, sampling,
  smoothing, and voltage window. [R7]

- The LLM should report ICA/DVA features as "peak drift", "peak broadening",
  "peak intensity change", or "peak disappearance" unless a validated
  chemistry-specific degradation map is supplied. [R6], [R7]

- Peak drift can be consistent with changes such as LLI or electrode active
  material loss, but assigning LLI/LAM_PE/LAM_NE requires chemistry-specific
  map support or additional diagnostics. [R6], [R7]

- Numerical derivatives amplify noise. If the input notes sparse sampling,
  irregular sampling, high C-rate, weak smoothing, or low peak prominence, the
  LLM should downgrade confidence in ICA/DVA-based conclusions. [R7]

- If `ica_peak_count` or `dva_peak_count` changes across aging, the LLM may
  describe that as a qualitative change in curve structure; it should not infer
  a specific electrode mechanism without map validation. [R6], [R7]

### 4. Relaxation and Rest Features

- Voltage relaxation after current interruption contains information useful for
  capacity estimation; Nature Communications 2022 showed that features derived
  from relaxation voltage profiles can estimate capacity across commercial
  lithium-ion cells and validation datasets. [R5]

- `rest_voltage_delta_v`, `rest_voltage_at_30s_v`, `rest_voltage_at_60s_v`,
  `rest_voltage_at_300s_v`, slope features, and exponential-fit parameters
  should be interpreted as relaxation-shape descriptors, not direct proof of a
  single physical process. [R5], [R11]

- Increasing `rest_exp_tau_s` over cycles can support a hypothesis of slower
  relaxation or increased transport/impedance limitation, but the LLM should
  rely on trend significance and should not use only first-vs-last comparison.
  [R5], [R13], [R14]

- Relaxation features are sensitive to SOC at rest, prior current, temperature,
  and rest duration. If these are not comparable, report the limitation before
  drawing capacity or resistance conclusions. [R5], [R11]

### 5. Field-Use and Stress Features

- Historical usage features such as average voltage, calendar time, current,
  temperature, SOC distribution, and other statistical summaries can support
  battery aging prediction from field data. Wang et al. used field data from a
  fleet of 60 electric buses operated for more than four years and extracted
  statistical features from usage behavior for aging prediction and
  uncertainty-aware prognosis. [R9]

- `high_soc_rest_fraction` should be described as exposure to high-SOC rest,
  not as automatic evidence of a specific degradation mode. The LLM should use a
  configured threshold or batch percentile and should mention the threshold
  source. [R9], [R12]

- `calendar_time_h` is important because battery degradation can include
  time-dependent aging in addition to cycling-induced aging. Interpret calendar
  exposure jointly with SOC, temperature, and rest conditions. [R9], [R10],
  [R12]

- `current_variance_a2` should not be compared directly across cells unless
  nominal capacity is the same. Prefer `c_rate_variance` or other
  nominal-capacity-normalized dynamic-current metrics. [R9], [R10]

- Dynamic discharge profiles can produce different lifetime outcomes than
  constant-current profiles at the same average current and voltage window.
  Geslin et al. found that realistic dynamic discharge increased lifetime
  (by up to 38% in equivalent full cycles at end of life) relative to
  constant-current cycling in their studied cells, and that interpretable ML
  (XGBoost with SHAP analysis) highlighted low-frequency current pulses and
  time-induced aging as important under realistic conditions. [R10]

- Do not assume that dynamic current always accelerates degradation. The effect
  depends on current profile, average C-rate, rest periods, chemistry, SOC
  window, and temperature. [R10]

- `low_frequency_current_power` may be relevant when interpreting dynamic
  current profiles, but it should be treated as an association feature unless a
  validated model connects it to the target lifetime metric for the same
  chemistry/protocol class. [R10]

### 6. Temperature Features

- Temperature features such as mean, maximum, integral, and histogram exposure
  are relevant because battery degradation depends on operating and storage
  environment; however, temperature effects are intertwined with current,
  internal resistance, SOC, and protocol. [R2], [R9], [R12]

- The LLM should avoid a simple "higher temperature always means worse" claim
  unless the comparison controls for chemistry, C-rate, SOC, and calendar time.
  State the observed temperature exposure and whether it is unusually high
  relative to the dataset or specification. [R9], [R12]

### 7. EIS and Impedance Features

- EIS features such as high-frequency intercept, semicircle width, charge-transfer
  related arcs, Warburg-like low-frequency behavior, and absolute impedance can
  provide information-rich, non-invasive evidence for SOH and RUL models. [R11]

- Zhang et al. combined EIS with Gaussian process machine learning to identify
  degradation patterns and forecast battery health, supporting EIS as a valuable
  diagnostic modality when available. [R11]

- Without EIS or pulse-test data, the LLM should not make strong claims about
  ohmic resistance, charge-transfer resistance, or diffusion impedance from
  voltage/current cycle summaries alone. [R6], [R11]

- DRT peak areas should only be interpreted if the upstream pipeline provides a
  validated DRT inversion method and frequency range; placeholder NaN values
  must be reported as unavailable. [R11]

### 8. Modeling, SHAP, and Statistical Interpretation

- ElasticNet, tree models, Gaussian processes, and other supervised models can
  use engineered battery features for cycle-life, SOH, or RUL prediction, but
  model validity is conditional on the training distribution, test protocol, and
  feature leakage controls. [R2], [R3], [R4], [R11]

- SHAP can identify which features influence a fitted model's prediction for a
  given dataset, but SHAP values are not electrochemical mechanism measurements.
  Use SHAP to prioritize evidence and questions, not to prove causality. [R3],
  [R10]

- If the target is lifetime prediction across different cycling protocols, some
  protocol-encoding features may be intentionally useful. If the target is
  protocol-blinded cell quality variation, protocol-encoding features can
  artificially inflate performance and reduce transferability. [R3]

- Report uncertainty whenever available: confidence intervals, p-values,
  cross-validation errors, prediction intervals, residuals, missing feature
  groups, and out-of-distribution warnings. [R3], [R9], [R11]

## Recommended LLM Output Structure

For every analyzed cell or batch, the LLM should produce:

1. **Observed feature facts**: list the measured feature trends and values without
   mechanism claims. [R2], [R9]

2. **Supported interpretations**: map features to plausible battery-health
   implications with confidence and references. [R6], [R7], [R11]

3. **Evidence limitations**: state missing diagnostics, weak sample sizes,
   protocol/chemistry mismatch, or unavailable EIS/relaxation/ICA data. [R3],
   [R7]

4. **Modeling implications**: suggest whether the features are suitable for SOH,
   RUL, anomaly detection, protocol comparison, or LLM summarization. [R2], [R3],
   [R9]

5. **Next measurements**: recommend targeted diagnostics such as repeated
   reference performance tests, low-rate diagnostic cycles, EIS, pulse tests, or
   controlled-temperature repeats when the current evidence is ambiguous. [R6],
   [R7], [R11]

## Example Evidence Statements

- "The normalized discharge capacity shows a statistically significant negative
  trend. This supports capacity fade under the tested protocol, but it does not
  uniquely identify LLI, LAM_PE, or LAM_NE without additional diagnostics."
  [R6], [R13], [R14]

- "Delta-Q(V) features changed between the reference and target early cycles.
  Similar early voltage-curve features have been shown to predict cycle life in
  LFP/graphite fast-charging datasets, but transfer to other chemistries or
  protocols should be validated." [R2], [R3]

- "The primary ICA peak shifted over cycles. This is evidence of evolving
  voltage-curve structure; mechanism assignment requires chemistry-specific
  degradation maps or corroborating diagnostics." [R6], [R7]

- "High-SOC rest exposure is above the configured threshold. This is a usage
  stress signal, not a standalone mechanism diagnosis." [R9], [R12]

- "The model's SHAP values emphasize low-frequency current features. This is
  consistent with recent dynamic-cycling work showing the importance of
  low-frequency current pulses, but SHAP remains associative for the fitted
  model." [R10]

## IEEE-Style References

[R1] C. R. Birkl, M. R. Roberts, E. McTurk, P. G. Bruce, and D. A. Howey,
"Degradation diagnostics for lithium ion cells," *Journal of Power Sources*,
vol. 341, pp. 373-386, 2017, doi: 10.1016/j.jpowsour.2016.12.011.

[R2] K. A. Severson *et al.*, "Data-driven prediction of battery cycle life
before capacity degradation," *Nature Energy*, vol. 4, pp. 383-391, 2019,
doi: 10.1038/s41560-019-0356-8.

[R3] A. Geslin, B. H. C. van Vlijmen, X. Cui, A. Bhargava,
P. A. Asinger, R. D. Braatz, and W. C. Chueh, "Selecting the appropriate
features in battery lifetime predictions," *Joule*, vol. 7, no. 9,
pp. 1956-1965, 2023, doi: 10.1016/j.joule.2023.07.021.

[R4] P. M. Attia, A. Grover, N. Jin, K. A. Severson, T. M. Markov,
Y.-H. Liao, M. H. Chen, B. Cheong, N. Perkins, Z. Yang, P. K. Herring,
M. Aykol, S. J. Harris, R. D. Braatz, S. Ermon, and W. C. Chueh,
"Closed-loop optimization of fast-charging protocols for batteries with
machine learning," *Nature*, vol. 578, pp. 397-402, 2020,
doi: 10.1038/s41586-020-1994-5.

[R5] J. Zhu, Y. Wang, Y. Huang, R. B. Gopaluni, Y. Cao, M. Heere,
M. J. Muehlbauer, L. Mereacre, H. Dai, X. Liu, A. Senyshyn, X. Wei,
M. Knapp, and H. Ehrenberg, "Data-driven capacity estimation of commercial
lithium-ion batteries from voltage relaxation," *Nature Communications*,
vol. 13, article 2261, 2022, doi: 10.1038/s41467-022-29837-w.

[R6] J. S. Edge, S. O'Kane, R. Prosser, N. D. Kirkaldy, A. N. Patel,
A. Hales, A. Ghosh, W. Ai, J. Chen, J. Yang, S. Li, M.-C. Pang,
L. B. Diaz, A. Tomaszewska, M. W. Marzook, K. N. Radhakrishnan, H. Wang,
Y. Patel, B. Wu, and G. J. Offer, "Lithium ion battery degradation: what
you need to know," *Physical Chemistry Chemical Physics*, vol. 23,
pp. 8200-8221, 2021, doi: 10.1039/D1CP00359C.

[R7] M. Dubarry and D. Ansean, "Best practices for incremental capacity
analysis," *Frontiers in Energy Research*, vol. 10, article 1023555, 2022,
doi: 10.3389/fenrg.2022.1023555.

[R8] P. K. Herring, C. Balaji Gopal, M. Aykol, J. H. Montoya,
J. Anapolsky, P. M. Attia, W. Gent, J. S. Hummelshoj, L. Hung,
H.-K. Kwon, P. Moore, D. Schweigert, K. A. Severson, S. Suram,
Z. Yang, R. D. Braatz, and B. D. Storey, "BEEP: A Python library for
Battery Evaluation and Early Prediction," *SoftwareX*, vol. 11, article
100506, 2020, doi: 10.1016/j.softx.2020.100506.

[R9] Q. Wang, Z. Wang, P. Liu, L. Zhang, D. U. Sauer, and W. Li,
"Large-scale field data-based battery aging prediction driven by
statistical features and machine learning," *Cell Reports Physical Science*,
vol. 4, no. 12, article 101720, 2023, doi: 10.1016/j.xcrp.2023.101720.

[R10] A. Geslin, L. Xu, D. Ganapathi, *et al.*, "Dynamic cycling enhances
battery lifetime," *Nature Energy*, vol. 10, pp. 172-180, 2025,
doi: 10.1038/s41560-024-01675-8.

[R11] Y. Zhang, Q. Tang, Y. Zhang, J. Wang, U. Stimming, and A. A. Lee,
"Identifying degradation patterns of lithium ion batteries from impedance
spectroscopy using machine learning," *Nature Communications*, vol. 11,
article 1706, 2020, doi: 10.1038/s41467-020-15235-7.

[R12] J. Vetter, P. Novak, M. R. Wagner, C. Veit, K.-C. Moeller,
J. O. Besenhard, M. Winter, M. Wohlfahrt-Mehrens, C. Vogler, and
A. Hammouche, "Ageing mechanisms in lithium-ion batteries," *Journal of
Power Sources*, vol. 147, no. 1-2, pp. 269-281, 2005,
doi: 10.1016/j.jpowsour.2005.01.006.

[R13] M. G. Kendall, *Rank Correlation Methods*, 4th ed. London, U.K.:
Griffin, 1975.

[R14] P. K. Sen, "Estimates of the regression coefficient based on Kendall's
tau," *Journal of the American Statistical Association*, vol. 63, no. 324,
pp. 1379-1389, 1968, doi: 10.1080/01621459.1968.10480934.

[R15] J. He, Z. Wei, X. Bian, and F. Yan, "State-of-health estimation of
lithium-ion batteries using incremental capacity analysis based on
voltage-capacity model," *IEEE Transactions on Transportation
Electrification*, vol. 6, no. 2, pp. 417-426, 2020,
doi: 10.1109/TTE.2020.2994543.
