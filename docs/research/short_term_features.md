# Short-term battery features for machine interpretation

Date: 2026-08-16

## Decision

For a short test that cannot support ageing or state-of-health trends, the smallest useful BFL portfolio is four protocol-agnostic record families:

1. `operation.window_summary` — a compact operating/load envelope.
2. `response.capacity_aligned_profile` — paired charge/discharge voltage, current, and polarisation on a common capacity coordinate.
3. `response.current_step` — local voltage response to an eligible current step or pulse.
4. `response.relaxation_signature` — voltage relaxation during an eligible rest after load.

These four records give downstream software complementary views of what the cell experienced, its charge/discharge terminal response, how it responded to rapid load changes, and how it recovered, without claiming ageing, chemistry, state of health, or a named protocol. If implementation must be reduced to three, defer the capacity-aligned profile because it requires a complete paired charge/discharge window; the other three remain useful on fragmented dynamic data.

The present example demonstrates why aggregation matters: its existing output has thousands of phase/mode records and more than a thousand rest records, while pulse resistance, ICA/DVA, and evolution are not computable. Event records should remain traceable, but each family also needs one compact aggregate with member record IDs and representative/extreme source intervals.

## 1. Operating/load envelope (`operation.window_summary`)

### Purpose

Describe the electrical and thermal conditions actually observed in a selected scope. This is the reference context for every response feature and is useful even when no complete cycle exists.

### Minimum content

- Scope duration; included and excluded duration; declared gaps when upstream evidence exists, otherwise explicitly named sampling-interval outliers.
- Charge/discharge/rest duration and fraction.
- Waveform-mode counts and durations (`constant_current_like`, `constant_voltage_like`, `pulse_like`, `dynamic_current`, `unmatched`).
- Time-weighted current and absolute-current RMS and q05/q50/q95/q99; C-rate equivalents only when a traceable capacity reference exists.
- Time-weighted voltage q05/q50/q95 plus observed minimum and maximum; distinguish an isolated single-sample extreme from a sustained extreme.
- Time-weighted power or absolute-power q50/q95/q99; charge/discharge capacity and energy throughput; `I_squared_time` as a clearly labelled electrical-stress exposure integral.
- Temperature q05/q50/q95/maximum and temperature coverage when a standardised temperature channel exists.
- Counts and IDs of current steps and meaningful rests; pulse amplitude/dwell quantiles; plus a small set of representative and extreme intervals.
- Optional high-absolute-current/high-temperature overlap only when both the current threshold/reference and a mapped temperature channel are explicit; describe it as co-exposure, not causality or safety risk.

### Method and QA

- Use previous-value zero-order-hold intervals (`x[i]` applies on `[t[i], t[i+1])`) for time-weighted statistics and integration. Do not weight irregular samples equally.
- Do not fill missing current, voltage, power, or temperature with zero.
- Exclude non-positive time deltas. Do not infer missing-data gaps from a global sampling median: keep isolated sampling-interval outliers in ZOH totals, report their count/duration, and use them only to gate checkpoint interpolation when temporal support is too sparse.
- Report both robust quantiles and extrema. An extremum alone must not be described as a sustained exposure.
- State the current-sign convention, capacity-reference source, temperature channel, and scope-selection rule in `reference_frame`.
- Treat observed voltage/temperature ranges as measurements, not safety-limit violations, unless explicit manufacturer limits are supplied.

### Evidence

- The Battery Data Toolkit separates source/column provenance, derived post-processing, and consistency warnings, which supports keeping method and quality with every derived summary: [metadata](https://rovi-org.github.io/battery-data-toolkit/user-guide/schemas/index.html), [post-processing](https://rovi-org.github.io/battery-data-toolkit/user-guide/post-processing/index.html), and [consistency checks](https://rovi-org.github.io/battery-data-toolkit/user-guide/consistency/index.html).
- A large *Scientific Data* battery dataset publishes run-level operating condition, rate, voltage/current limits, charge/energy throughput, efficiency, OCV, temperature, and duration together. It also warns that dynamic data at coarse resolution can underestimate throughput, gaps should not be naively interpolated, and efficiency is invalid for asymmetric or too-short operations: [doi:10.1038/s41597-024-03831-x](https://doi.org/10.1038/s41597-024-03831-x).

## 2. Capacity-aligned terminal profile (`response.capacity_aligned_profile`)

### Purpose

Describe charge/discharge voltage separation and current on a common electrochemical progress coordinate. Unlike a single capacity or efficiency number, the profile tells a downstream consumer where terminal polarisation is small or large across the observed window.

### Minimum content

- The charge and discharge source intervals, structural-completeness decision, phase directions, temperatures, and median currents.
- A fixed common coordinate with `charge_voltage`, `discharge_voltage`, `charge_current`, and `discharge_current` series.
- `voltage_gap = charge_voltage - discharge_voltage` on that coordinate.
- Optional `apparent_paired_polarisation = voltage_gap / (charge_current - discharge_current)` where the current difference is safely non-zero.
- Compact summaries: mean charge/discharge voltage (`E/Q`), median/q95 voltage gap, integrated voltage gap, and the coordinate/interval of maximum sustained gap.

The coordinate name must match its evidence:

- `normalized_capacity_fraction` only when a traceable nominal/reference capacity exists;
- `aligned_capacity_ah` for a complete pair compared over a common absolute-capacity overlap;
- never label a locally normalised partial phase as SOC.

### Method and QA

- Require structurally complete, paired charge and discharge phases with compatible endpoints or a declared common capacity window.
- Require monotone capacity, high finite coverage, sufficient common overlap, and no interpolation across a declared gap or a locally unsupported sampling interval.
- Interpolate each phase only inside its observed capacity support and record grid spacing and interpolation method.
- Record current and temperature comparability. Do not interpret a voltage gap as an inherent cell property when charge/discharge rates or temperatures differ materially.
- Apply a declared denominator floor before calculating paired polarisation.
- Call `apparent_paired_polarisation` a terminal response, not intrinsic/internal resistance.
- Do not transfer any lifetime or health prediction from the source paper.

Zhang et al. align charge/discharge voltage and current by normalised capacity to form `Vc(Q)`, `Vd(Q)`, `Ic(Q)`, and `Id(Q)`, then derive `delta V(Q) = Vc(Q) - Vd(Q)` and `R(Q) = delta V(Q)/(Ic(Q) - Id(Q))`: [Nature Machine Intelligence version of record, doi:10.1038/s42256-024-00972-x](https://doi.org/10.1038/s42256-024-00972-x) and the [open Methods manuscript](https://doi.org/10.21203/rs.3.rs-3718134/v1). Their validated target is lifetime prediction over early cycles; BFL should adopt only the capacity-aligned terminal representation and its explicit reference coordinate.

Battdat independently computes charge/discharge capacity from current and energy from current times voltage, but explicitly notes that cycle capacity/energy assumes return to the starting state. This supports a hard completeness gate: [Battdat `CapacityPerCycle`](https://rovi-org.github.io/battery-data-toolkit/source/postprocess.html#battdat.postprocess.integral.CapacityPerCycle).

## 3. Current-step response (`response.current_step`)

### Purpose

Capture a short-timescale terminal response without requiring a complete cycle. This is the highest-value addition for dynamic identification data and remains useful when SOC is unavailable.

### Minimum event content

- Direction and surrounding phases; event start/end and source rows/records.
- Pre-step baseline voltage/current, pulse plateau current, `delta_current`, pulse duration, and actual sample latency.
- `delta_voltage_first_valid`, `apparent_dc_resistance_first_valid`, and, when covered, the same quantities at declared times such as 1, 2, 5, and 10 s.
- `polarization_growth = R_10s - R_first_valid` when both values are valid.
- Post-pulse voltage recovery at declared times when a following rest exists.
- Pre-step voltage and temperature as the always-available state coordinates; SOC only when its reference is explicit and traceable.
- One aggregate grouped by direction and coarse pre-voltage/temperature bins, containing event count, median/q05/q95 responses, member IDs, and outlier IDs.

Use the neutral name `apparent_dc_resistance` when no controlled HPPC/GITT protocol or equilibrated OCV is established. Do not call it intrinsic resistance or assign it to a particular electrochemical mechanism.

### Provider branches

- If a valid SOC reference and PyProBE-required columns exist, use `pyprobe.analysis.pulsing.get_resistances`. PyProBE defines OCV as the last zero-current voltage before the pulse, R0 from that OCV and the first pulse sample whose current is within 1% of the median pulse current, and optional `R_t` at requested times: [immutable PyProBE 2.6.0 source](https://github.com/ImperialCollegeLondon/PyProBE/blob/v2.6.0/pyprobe/analysis/pulsing.py#L74-L203), [pulse API](https://pyprobe.readthedocs.io/en/latest/_autosummary/pyprobe.analysis.pulsing.html), and [worked example](https://pyprobe.readthedocs.io/en/latest/examples/analysing-GITT-data.html).
- If SOC is absent, a separately named BFL `current_step_delta_v_over_delta_i_v1` branch may emit the local apparent response. This is a predeclared method, not a same-name fallback for a failed PyProBE call. SOC remains `not_computable`.
- BEEP is useful precedent, but not a generic provider here: its `HPPCResistanceVoltageFeatures` first validates a named HPPC cycle and extracts resistance at 0 s, 3 s, and pulse end across SOC windows: [BEEP source at commit f2b984e](https://github.com/TRI-AMDD/beep/blob/f2b984e4bbf54554d9514c1defc81d09ac2a3dee/beep/features/core.py) and [helper definitions](https://github.com/TRI-AMDD/beep/blob/f2b984e4bbf54554d9514c1defc81d09ac2a3dee/beep/features/featurizer_helpers.py).

### Eligibility and QA

- Stable pre-step baseline with enough samples and duration; record the exact baseline window.
- Non-trivial `delta_current` relative to observed current noise/resolution.
- Pulse plateau reaches and maintains the declared current; record current coefficient of variation.
- Requested response time lies inside the observed pulse and sampling resolution is adequate; record actual measurement/interpolation time.
- No non-positive time interval, locally unsupported sampling interval, voltage/current saturation, or truncated/current-limited pulse in the calculation interval.
- Keep charge and discharge events separate and retain the canonical current-sign convention.
- No resistance comparison across tests unless timescale, current amplitude/direction, pre-voltage or SOC, and temperature are comparable.

The USABC/INL test manual uses `delta V / delta I` at 2 s and 10 s and says resistance should normally be calculated only for full-duration, full-amplitude pulses: [FreedomCAR Battery Test Manual](https://inldigitallibrary.inl.gov/content/uploads/50/2026/04/6308373.pdf). Barai et al. show experimentally that pulse resistance depends strongly on measurement timescale and that instantaneous resistance is limited by acquisition rate: [doi:10.1038/s41598-017-18424-5](https://doi.org/10.1038/s41598-017-18424-5). These points require recording timescale and acquisition latency as first-class fields.

## 4. Relaxation signature (`response.relaxation_signature`)

### Purpose

Describe recovery of terminal voltage after charge, discharge, or a pulse. It complements the loaded response and does not require a long ageing series.

### Minimum event content

- Preceding phase/mode, preceding median/end current, load duration, and rest duration.
- Start/end voltage and signed `delta_voltage`.
- `delta_voltage` at fixed observed checkpoints (recommended MVP: 10, 30, 60, and 300 s; add 600 and 1800 s when observed); omit rather than extrapolate checkpoints beyond the interval.
- Time-weighted mean and variance of voltage; maximum/minimum and monotonic fraction. For rests with enough distinct samples, optional robust early/late slopes on explicitly declared time windows.
- Direction vocabulary: `recovering_up`, `recovering_down`, or `nonmonotone`.
- Temperature baseline/change over the same interval when available.
- One aggregate separated by preceding charge/discharge/pulse, with counts, duration distribution, checkpoint-response distributions, member IDs, and representative/outlier intervals.

### Eligibility and QA

- Rest current stays within the declared threshold for the complete interval.
- Rest is preceded by an observed non-rest interval; never infer missing prehistory.
- At least two finite voltage samples for checkpoints; require at least ten distinct samples for curve-shape moments or slopes, sufficient duration for each emitted metric, no declared gap or locally unsupported sampling interval, and adequate finite-data coverage.
- Record interpolation method and bracketing sample times. Do not extrapolate.
- Call the value `terminal_relaxation_voltage`, not OCV or equilibrium voltage, unless a separate equilibrium criterion is declared and met.
- A single relaxation curve must not produce a capacity, SOH, lithium-plating, or mechanism diagnosis.

Zhu et al. converted post-charge relaxation curves into variance, skewness, maximum, minimum, mean, and excess kurtosis and found `[variance, skewness, maximum]` most effective in their trained capacity-estimation setting: [doi:10.1038/s41467-022-29837-w](https://doi.org/10.1038/s41467-022-29837-w). BFL should adopt only the descriptive evidence (fixed-time response and sample-rate-aware curve statistics), not their trained capacity mapping. Their paper explicitly shows dependence on cell type and cycling condition, so uncalibrated inference would be unsupported.

## Existing gated ICA/DVA path to retain (not new scope)

### Purpose

The existing ICA/DVA path should remain available when an eligible slow constant-current charge or discharge exists, but it is not one of the recommended new records. It will correctly be unavailable for many dynamic short tests.

### Minimum content

- Phase direction, median current/C-rate, temperature, voltage span, capacity span, and source interval.
- Full processed ICA (`dQ/dV`) and DVA (`dV/dQ`) series for numerical consumers.
- For compact consumers: peak/trough voltage or capacity coordinate, signed magnitude, prominence, width when stable, and ranked landmark IDs.
- Filtering, differentiation, uniform-section choice, and peak-detection parameters.

### Provider and QA

- Use PyProBE `differentiate_lean`, which implements the LEAN method, assumes evenly spaced x data, and exposes the bin multiple, smoothing coefficients, and constant-sampling section: [PyProBE differentiation API](https://pyprobe.readthedocs.io/en/latest/_autosummary/pyprobe.analysis.differentiation.html).
- Cellpy independently exposes `dqdv`/`dqdv_cycle`/`dqdv_np`, reinforcing ICA as an ecosystem-standard analysis, but its different interpolation and smoothing settings mean results are not interchangeable unless parameters match: [cellpy ICA documentation](https://cellpy.readthedocs.io/en/latest/examples/04_incremental_capacity_analysis.html).
- Require a slow constant-current phase, sufficient point count and voltage span, high finite coverage, monotone capacity and voltage, and a traceable capacity reference before applying a C-rate gate.
- Peak positions may describe the curve. Assigning chemistry, degradation modes, or health changes requires chemistry/reference-electrode information or a comparable baseline and is outside this short-test record.

The underlying differentiation method is documented by Feng et al.: [doi:10.1016/j.etran.2020.100051](https://doi.org/10.1016/j.etran.2020.100051). PyProBE itself is peer-reviewed at [doi:10.21105/joss.07474](https://doi.org/10.21105/joss.07474), and cellpy at [doi:10.21105/joss.06236](https://doi.org/10.21105/joss.06236).

## Minimal JSON contract additions

Keep the existing record envelope. Add structured reference and confidence fields inside `attributes` rather than generated prose:

```json
{
  "record_type": "response.current_step",
  "source_intervals": [{"start_record": 100, "end_record": 145, "start_time_s": 20.0, "end_time_s": 40.0}],
  "attributes": {
    "direction": "discharge",
    "preceding_phase": "rest",
    "reference_frame": {
      "voltage_baseline": "median_of_last_stable_pre_step_window",
      "current_sign": "charge_positive",
      "state_axis": "pre_step_voltage",
      "soc_reference": null
    },
    "confidence": {
      "level": "high",
      "basis": ["stable_baseline", "full_amplitude", "time_bracketed"],
      "not_a_probability": true
    }
  },
  "metrics": {
    "delta_current": {"value": -35.2, "unit": "A", "status": "ok", "reason": null},
    "apparent_dc_resistance_10s": {"value": 0.0031, "unit": "ohm", "status": "ok", "reason": null},
    "soc": {"value": null, "unit": "1", "status": "not_computable", "reason": "no traceable SOC reference"}
  },
  "method": {
    "provider": "BFL",
    "name": "current_step_delta_v_over_delta_i_v1",
    "provider_version": "0.4.x",
    "parameters": {"response_times_s": [1.0, 2.0, 5.0, 10.0]},
    "references": ["https://doi.org/10.1038/s41598-017-18424-5"]
  },
  "applicability": {"status": "applicable", "reasons": []},
  "quality": {"status": "ok", "flags": []},
  "interpretation_limits": ["A local terminal response; not an intrinsic or health-specific resistance."]
}
```

Confidence should be categorical and rule-derived (`high`, `medium`, `low`, `not_computable`), with explicit basis and failed gates. An uncalibrated numeric confidence score would imply a statistical meaning that BFL has not established.

## Explicitly defer

- Capacity retention, fade slope, SOH, remaining useful life, or significant degradation from a single/short test.
- Chemistry or named-protocol classification from waveform shape alone.
- SOC inferred from voltage or filenames without a validated chemistry-specific reference.
- RC time constants, diffusion coefficients, or mechanistic labels from uncontrolled rests/pulses.
- Lithium-plating or safety diagnosis from a single relaxation/ICA landmark.
- Coulombic/energy efficiency or hysteresis from incomplete, asymmetric, or state-mismatched charge/discharge windows.

## Suggested delivery order

1. Extend the existing exposure record into one `operation.window_summary` and add compact mode/phase aggregation.
2. Add `response.current_step` with the PyProBE branch when SOC is valid and a separately named local apparent-response branch when it is not.
3. Refine rest output into meaningful `response.relaxation_signature` events plus one aggregate; avoid emitting every trivial zero-current fragment as an equally important record.
4. Add `response.capacity_aligned_profile` only for a complete, comparable charge/discharge pair.
5. Keep the current PyProBE ICA/DVA implementation as optional; do not add a second provider or duplicate algorithm.

This is sufficient for downstream software to form grounded descriptions while retaining the source interval, derivation, reference frame, applicability, quality evidence, and interpretation limits for every statement.
