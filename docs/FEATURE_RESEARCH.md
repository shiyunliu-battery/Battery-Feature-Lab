# Battery Feature Extraction — Literature-Grounded Methodology

> Purpose: a reference for **which features to extract, which statistical methods to use, and how
> to avoid hard-coded assumptions**, so that Battery Feature Lab is publishable and reusable.
> Every feature family below is traced to peer-reviewed work (Nature Energy / Nature Communications
> / Joule / PNAS / J. Power Sources). This document is methodology, not implementation.

## Design principles (publishable / no hard-code)

1. **No bare magic constants for decisions.** Every threshold must come from one of:
   (a) the cell specification injected via config (e.g. nominal capacity, max C-rate),
   (b) the data distribution itself (percentiles, z-scores, change-point detection), or
   (c) a statistical significance test (p-value).
2. **Trend decisions use statistical tests**, not slope-vs-constant comparisons. Use the
   non-parametric **Mann-Kendall** trend test (+ **Theil-Sen / Sen's slope** estimator), the
   standard in geosciences/econometrics for monotonic trend detection.
3. **Normalize every curve feature** to beginning-of-life (BOL) or nominal, so features are
   comparable across cells and chemistries.
4. **Degradation-mode outputs are evidence, not verdicts**: emit `{evidence, possible_modes,
   confidence}` triggered by tests, never a hard yes/no label.
5. **Tag every feature row with the operating condition** it was extracted under (C-rate,
   temperature, SOC window) — otherwise features are not comparable across protocols.

---

## 1. Feature families and their literature sources

### 1.1 ΔQ(V) early-life curve features — Severson 2019, *Nature Energy*
Foundational. Interpolate cycle-100 and cycle-10 discharge capacity onto a common **voltage grid**,
take the difference ΔQ_{100-10}(V). The **variance** of ΔQ(V) correlates with log(cycle life) at
R²≈0.86; single-feature ~15% error, full model 9.1%, classification from 5 cycles at 4.9%.

- Features: `var`, `min`, `mean`, `skewness`, `kurtosis` of ΔQ(V); area, |area|, L1, L2.
- Physics: larger variance ⇒ more non-uniform energy dissipation vs voltage ⇒ faster fade.
- Status in code: implemented in `featurizers/delta_q.py` (log-variance present). **To add:**
  skewness, kurtosis (used in original work).
- Severson et al., Nature Energy 4, 383–391 (2019). https://www.nature.com/articles/s41560-019-0356-8
  · PDF: https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf

### 1.2 ICA / DVA (dQ/dV, dV/dQ) — Dubarry, Bloom, Weng et al.
Strongest curves for **degradation-mode** diagnosis; map directly to electrochemistry.

| Observation | Mechanism |
|---|---|
| Peak **area** decreases | LAM (loss of active material); area ∝ reversible capacity of phase |
| Peak **position** shifts | LLI (loss of lithium inventory) / electrode slippage |
| Peak **height / FWHM** change | kinetics, polarization, ORI (ohmic resistance increase) |

- Features per peak: position(V), height, area, FWHM, peak count, inter-peak spacing.
- Engineering requirement: dQ/dV is noise-sensitive — **resample onto a uniform V/Q grid, then
  Savitzky-Golay smooth, then differentiate**. (Implemented via `safe_savgol`+`safe_gradient`.)
- Rate dependence: ICA peaks shift with C-rate ⇒ **store the extraction C-rate per row**, else
  cross-protocol comparison is invalid.
- Peak-tracking degradation modes, J. Energy Storage (2021):
  https://www.sciencedirect.com/science/article/abs/pii/S2352152X2101344X
  · ICA-DV generation review, Energies 17, 4309 (2024): https://www.mdpi.com/1996-1073/17/17/4309

### 1.3 Relaxation-voltage features — Zhu et al. 2022, *Nature Communications*
Easy to acquire, independent of cycling history. After full charge, from the rest/relaxation
voltage curve extract **6 statistical features**: `Var`, `Skewness`, `Excess-Kurtosis`, `Max`,
`Min`, `Mean` (skew/kurt normalized by variance).

- Physics: as capacity fades, the relaxation-voltage distribution sharpens ⇒ variance decreases.
- Status: `featurizers/relaxation.py` already fits exponential τ + moments. **To do:** align the
  6-feature naming to Zhu 2022 for direct reproducibility/citation.
- Zhu et al., Nature Communications 13, 2261 (2022):
  https://www.nature.com/articles/s41467-022-29837-w

### 1.4 EIS / DRT impedance features — Zhang et al. 2020, *Nature Communications* + DRT
Two complementary routes:

(a) **Direct spectral features (Zhang 2020):** 20,000+ EIS spectra + Gaussian process; specific
frequency points of Im(Z)/Re(Z) are themselves strong health indicators — no equivalent-circuit
fit required. Features: Z_real, Z_imag, |Z|, phase at selected frequencies.

(b) **DRT distribution (model-free, recommended):** deconvolve EIS into the time-constant domain;
each **peak area = the resistance of that process**:
- τ < 10⁻⁵ s → ohmic resistance
- 10⁻⁵–10⁻³ s → SEI ion transport
- 10⁻³–10⁻¹ s → charge transfer (Rct)
- τ > 10⁻¹ s → solid-state diffusion

DRT avoids the equivalent-circuit risk of mis-attributing resistance growth as the cell ages.
Features: per-τ-band peak area, position, height vs aging.

- Zhang et al., Nature Communications 11, 1706 (2020):
  https://www.nature.com/articles/s41467-020-15235-7
  · DRT aging analysis, Batteries 11, 34 (2025): https://www.mdpi.com/2313-0105/11/1/34

### 1.5 Differential thermal voltammetry (DTV) + Coulombic efficiency — advanced/optional
- **DTV (dT/dV):** needs only voltage + temperature; peaks correspond to electrode phase-transition
  entropic heat; peak parameters quantify SOH in operando.
- **High-precision Coulombic efficiency (Dahn group):** deviation of CE from 1 predicts life very
  early. Report `1 - CE` (more linear). `CE = Q_discharge / Q_charge`.
- DTV, J. Power Sources (2014):
  https://spiral.imperial.ac.uk/server/api/core/bitstreams/f23e1c5e-5e1e-406e-b0bb-af895187384b/content

### 1.6 Stress / usage features + path dependence — Attia knee review
- SOC/voltage/current/temperature histograms, DOD, duty cycle (already implemented).
- **Add rainflow counting** (standard fatigue-analysis algorithm) to quantify the DOD spectrum —
  the canonical way to capture path-dependent degradation.
- Temperature exposure via **Arrhenius weighting** (∫ exp(−Ea/RT) dt) rather than a plain mean.
- Knee review & pathways: https://iopscience.iop.org/article/10.1088/3049-4761/ae5067
  · Knee-onset detection, PNAS (2024): https://www.pnas.org/doi/10.1073/pnas.2424838122

### 1.7 Knee point / degradation-trajectory features
- knee-onset = transition from quasi-linear to accelerated fade; earlier warning than the knee.
- Detection (**non-threshold**): Bacon-Watts two-segment regression, max of second difference, or
  the SOC-dependent parameter method (PNAS 2024).
- Features: knee-onset cycle, the two segment slopes, peak curvature.

---

## 2. Statistical methods (also evidence-based)

### 2.1 Distribution descriptors (apply uniformly to every curve)
mean, std, var, **skewness, kurtosis**, min, max, range, quantiles (Q10/Q25/median/Q75/Q90),
IQR, L1/L2 norms, area-under-curve. `featurizers/common.describe_array` covers most; add skew/kurt.

### 2.2 Feature selection (a publishable pipeline needs a layered approach)
| Method | Role | Note |
|---|---|---|
| Pearson | linear correlation | battery data is often non-linear |
| **Spearman** | monotonic, outlier-robust | better fit for battery data |
| **Mutual Information** | any non-linear/non-monotonic dependence | recommended primary |
| **VIF** | multicollinearity, drop if >5–10 | ICA features are highly collinear |
| **mRMR** | max-relevance / min-redundancy | resolves redundant high-correlation clusters |
| RFE / LASSO / **Elastic Net** | embedded selection | **Severson 2019 used Elastic Net** |

`analysis/feature_selection.py` already has correlation/MI/VIF. **To add:** mRMR and Elastic-Net
coefficients (aligns with Severson). Comprehensive feature-extraction review, J. Power Sources
(2026): https://www.sciencedirect.com/science/article/pii/S0378775326001394

### 2.3 SHAP — clarification
SHAP is a **model-explanation** method, not a feature. Pipeline: extract features → train a
SOH/RUL model → explain with **TreeSHAP** which features drive predictions → feed the SHAP ranking
+ feature values to the LLM for domain analysis. `analysis/shap_report.py` is correctly positioned
as an optional adapter. **Emit both** global importance and per-cell **local** SHAP (most valuable
for "LLM + domain knowledge explains a single cell"). Interpretable battery aging (SHAP), ChemRxiv:
https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/644b044280f4b75b533b1c9d/original/interpretable-data-driven-modeling-reveals-complexity-of-battery-aging.pdf

---

## 3. Removing hard-coded assumptions (publish-readiness)

Current highest-risk constants live in `analysis/degradation_tags.py`:

| Current (hard-coded) | Problem | Evidence-based replacement |
|---|---|---|
| `slope < -1e-4` fade trigger | arbitrary, unit-dependent | **Mann-Kendall** test (p-value + Sen's slope); tag only if p<0.05, report the slope |
| `high_soc_rest > 0.2` | where does 0.2 come from? | config parameter + documented default; or batch-distribution percentile |
| `current_variance > 1.0` | unit-dependent (A²) | normalize to C-rate via nominal capacity first; threshold → z-score/percentile |
| `max_discharge_c > 3.0` | chemistry-specific | inject `max_c_rate` from the datasheet; judge relatively |
| `tau[-1] > tau[0]*1.5` | arbitrary 1.5×, endpoints only | Mann-Kendall / regression significance over the full series |
| `len(discharge) < 25` (delta_q) | magic number | route through `config.min_points_for_curve` |
| ICA `prominence = std*0.5` | arbitrary 0.5 | relative noise level (MAD-based) or configurable |

The existing `_tag(possible_modes, confidence, evidence)` structure is the right shape — only the
trigger conditions need to become statistical tests. Recommend stating this methodology in the
README.

---

## 4. Recommended priority order

1. **Remove hard-codes first** (must-do before release): replace `degradation_tags` thresholds with
   Mann-Kendall + config injection. Only needs `scipy.stats`, no heavy new dependency.
2. **Align naming to papers:** relaxation → Zhu 2022 six features; ΔQ(V) → add skew/kurt (Severson).
3. **Add operating-condition metadata** per feature row: C-rate, temperature, SOC window.
4. **Add mRMR + Elastic Net** to feature selection; **add local SHAP** explanations.
5. **Advanced:** rainflow (stress), Bacon-Watts knee detection, real DRT (currently a placeholder).

---

## 5. Implemented in this iteration + open robustness notes

Done and verified (`tests/test_pipeline.py`, `scripts/validate_on_dataset.py`):

- **Single-cell-capable stress tags.** `high_soc_rest_exposure` now fires via an absolute,
  config-injected threshold (default documented as a conservative convention) so it works on a
  single cell, with the batch percentile retained only as a supplementary *relative-outlier*
  signal. The evidence string names which criterion fired. The "high SOC" level itself is now a
  config (`FeatureConfig.high_soc_level`, default 0.8, grounded in calendar-aging literature),
  removing the former hard-coded `soc >= 0.8`.
- **Datasheet-relative max C-rate** tag fires only when the cell spec is supplied (correct: an
  absolute physical limit must come from domain knowledge, not the data).
- **numpy 1.x/2.x compatibility.** `np.trapezoid` (numpy≥2.0 only) is wrapped by
  `core.integration.trapezoid`, so the pipeline runs across the declared `numpy>=1.23` range.
- **Validation harness.** `validation/dataset_validation.py` + `scripts/validate_on_dataset.py`
  reproduce the Severson criterion (log(var ΔQ(V)) vs log(cycle life)) and rank early-life
  features by Spearman correlation with life. Run the offline self-test with
  `python scripts/validate_on_dataset.py --synthetic 12`; point at real data with `--data-dir`
  (MATR/Severson via https://data.matr.io/1/, BEEP via https://github.com/TRI-AMDD/beep).

Robustness fixes applied:

- **Robust batch outlier (fixed).** The batch C-rate-variance criterion now uses
  **median + k·MAD** (k=3, MAD scaled by 1.4826), replacing the non-robust mean+kσ rule whereby a
  single extreme cell inflated σ and masked itself. Regression-tested with the exact masking case
  (`test_robust_mad_outlier_not_masked_by_single_extreme`).
- **Mann-Kendall sample size (fixed).** `min_trend_points` raised 4→8 (MK has low power below
  ~n=8–10). The docstring now notes that the test is Kendall's tau between the series and its cycle
  index — i.e. `scipy.stats.kendalltau` computes the Mann-Kendall test directly (exact for small n,
  asymptotic with tie handling otherwise), paired with the Theil-Sen (Sen's slope) estimator.

Still open:

- **Empirical validation gap.** The synthetic self-test confirms the *machinery*; scientific
  validation still requires running `--data-dir` on MATR/BEEP and confirming R² in the
  literature range (~0.6–0.86), not the self-test's degenerate R²≈1.0.

## Key references

- Severson et al., *Nature Energy* 4, 383 (2019) — ΔQ(V) variance, early-life life prediction.
- Zhu et al., *Nature Communications* 13, 2261 (2022) — relaxation-voltage 6 statistical features.
- Zhang et al., *Nature Communications* 11, 1706 (2020) — EIS + Gaussian process degradation.
- Attia et al. — knees in Li-ion aging trajectories; knee-onset detection (PNAS 2024).
- Dubarry / Bloom / Weng — ICA/DVA degradation-mode quantification (LLI/LAM/ORI).
- Comprehensive feature-extraction review, *J. Power Sources* (2026).
