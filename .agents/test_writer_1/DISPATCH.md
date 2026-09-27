# Dispatch Assignment — E2E Test Writer 1 (E2E Testing Track)

## Context
You are `test_writer_1` on the Dual-Site Blood Pressure Estimation Framework project.
Working directory: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\test_writer_1`
Project root: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`
Authoritative Request: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`
Project Index: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md`

## Mandatory Reading
Read `ORIGINAL_REQUEST.md`, `PROJECT.md`, and Explorer handoffs:
- `.agents/explorer_survey_1/handoff.md`
- `.agents/explorer_survey_2/handoff.md`
- `.agents/explorer_survey_3/handoff.md`

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Mission: Build E2E Test Infrastructure & Opaque-Box Suite
1. Create `TEST_INFRA.md` at project root following the dual-track testing principles:
   - 4-Tier methodology: Tier 1 (Feature coverage >= 5 per feature across all 15 features), Tier 2 (Boundary & Corner cases >= 5 per feature), Tier 3 (Cross-feature combinations pairwise >= 15), Tier 4 (Real-world clinical & mobile workloads >= 8). Total >= 170 test cases.
2. Build synthetic physiological waveform generators:
   - Generating dual-stream optical pulses (proximal face rPPG + distal finger PPG) with controlled pulse transit time $\Delta PTT$, respiratory Traube-Hering-Mayer modulation (~0.2 Hz), hydrostatic elevation offsets ($\Delta P = \rho g \Delta h$), and realistic sensor noise / VFR timestamp jitter.
3. Build the Opaque-Box E2E test suites in `tests/e2e/`:
   - `test_tier1_feature.py`: Isolated feature tests for optical extraction, SENSOR_TIMESTAMP PCHIP sync, sub-frame tangent foot detection, ePWV calculation, hydrostatic tilt perturbation, THM compliance, stiffness parameter identification ($E_0, \alpha$), DS-TCN model forward pass, tripartite loss, and INT8 quantization.
   - `test_tier2_boundary.py`: Boundary and corner cases (extreme bradycardia 35 BPM, extreme tachycardia 190 BPM, extreme hypertension SBP 210 / DBP 130, severe hypotension SBP 75 / DBP 45, short height 1.3m, tall height 2.1m, micro-elevation delta h = 10mm, macro-elevation delta h = 600mm, severe 20 dB noise, zero-tilt edge case).
   - `test_tier3_pairwise.py`: Combinations (high HR + hydrostatic perturbation; low compliance + high noise; VFR jitter + sub-frame foot detection; etc.).
   - `test_tier4_workloads.py`: Realistic 10-second acquisition runs simulating clinical subjects (MIMIC-III arterial line benchmarks, resting vs post-exercise state transitions matching MCD-Iriun format).
4. Build the Diagnostic Template Collapse Audit in `tests/audit/template_collapse_audit.py`:
   - Evaluates predictions across diverse validation cohorts.
   - Computes Pearson correlation $r$ (asserts $r \ge 0.82$).
   - Computes prediction spread variance ratio $\sigma^2(\hat{y}) / \sigma^2(y)$ (asserts $\ge 0.70$ and $\le 1.30$, strictly ruling out variance vanishing).
   - Computes Bland-Altman agreement limits and AAMI/ISO 81060-2 / BHS Grade A criteria (SBP MAE $\le 5.0$, DBP MAE $\le 3.5$, SD $\le 7.0$ mmHg).
5. Build the master test runner script `run_e2e_tests.py` that executes all tiers and outputs structured pass/fail summaries.
6. Verify your tests by running `pytest tests/e2e` and `python tests/audit/template_collapse_audit.py`.
7. When the test infrastructure and all test cases are fully created and passing against simulated ground truth, publish `TEST_READY.md` at project root.
8. Maintain `progress.md` with timestamps and deliver `handoff.md`.

## 2026-09-23T18:16:31Z
You are test_writer_1 (E2E Testing Architect & Suite Writer).
Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\test_writer_1
Read C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md, C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md, and C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\test_writer_1\DISPATCH.md.
Review Explorer handoffs in .agents/explorer_survey_1, 2, and 3.
MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work.
Implement TEST_INFRA.md at project root.
Build synthetic physiological waveform generators with ground truth.
Build complete opaque-box E2E test suites in tests/e2e/ (Tier 1 Feature coverage, Tier 2 Boundary/Corner cases, Tier 3 Pairwise combinations, Tier 4 Real-world workloads) and tests/audit/template_collapse_audit.py.
Build master runner run_e2e_tests.py.
Run the tests, verify they execute and pass, publish TEST_READY.md at project root.
Maintain progress.md and deliver handoff.md. Notify orchestrator via send_message.
