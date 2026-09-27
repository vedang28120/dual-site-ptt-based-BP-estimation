# Dispatch Assignment — Worker M2 (Milestone 2: Physics-Informed Hemodynamic Engine HRVP-PINN)

## Context
You are `worker_m2` on the Dual-Site Blood Pressure Estimation Framework project.
Working directory: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m2`
Project root: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`
Authoritative Request: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`
Project Index: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md`

## Mandatory Reading
Read `ORIGINAL_REQUEST.md`, `PROJECT.md`, and Explorer handoffs:
- `.agents/explorer_survey_3/handoff.md` (Crucial for sub-frame tangent foot detector, D_path, ePWV, IMU hydrostatic tilt, THM Hilbert envelope compliance, closed-form E0/alpha identification)

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Exclusive Write Ownership
You exclusively own:
- `src/pinn/`
  - `src/pinn/__init__.py`
  - `src/pinn/foot_detector.py`
  - `src/pinn/anthropometrics.py`
  - `src/pinn/hydrostatic_engine.py`
  - `src/pinn/respiratory_envelope.py`
  - `src/pinn/stiffness_solver.py`
- `tests/unit/test_pinn.py`

DO NOT modify files outside your ownership.

## Mission & Specifications
1. Implement `foot_detector.py`:
   - Sub-frame continuous intersecting tangent foot detection on dual optical waveforms (face rPPG vs finger contact PPG).
   - Fits continuous cubic spline / Savitzky-Golay polynomial over cardiac upstroke.
   - Solves for maximum upstroke slope tangent $L_2(t)$ ($S''(t)=0$) and diastolic baseline tangent $L_1(t)$ ($y=y_{\min}$).
   - Analytical closed-form intersection root $t_{\text{foot}} = t_{\text{max\_slope}} - \frac{y_0 - y_{\min}}{m}$.
   - Delivers $\le 0.75\text{ ms}$ precision (bypassing 33.3 ms integer sampling error).
2. Implement `anthropometrics.py`:
   - Anatomical path length: $D_{\text{path}} = 0.55 \cdot \text{Height} + 0.12 \cdot \text{ArmSpan}$ (meters). If ArmSpan unprovided, default to $1.03 \cdot \text{Height}$.
   - Pulse Wave Velocity: $\text{ePWV} = D_{\text{path}} / \Delta PTT$.
3. Implement `hydrostatic_engine.py`:
   - IMU hydrostatic perturbation: $\Delta P_{\text{hydro}} = \rho_{\text{blood}} \cdot g \cdot \Delta h$.
   - $\rho_{\text{blood}} = 1055\text{ kg/m}^3, g = 9.80665\text{ m/s}^2 \implies \Delta P_{\text{hydro}} = 0.077601 \times \Delta h\text{ [mm]}$ mmHg.
   - ZUPT zero-velocity integration for elevation displacement $\Delta h$.
4. Implement `respiratory_envelope.py`:
   - Traube-Hering-Mayer envelope extraction ($\sim 0.15 - 0.35\text{ Hz}$, centered at $0.2\text{ Hz}$) from cycle-by-cycle ePWV series.
   - Resamples to uniform 10 Hz via PCHIP, applies zero-phase Butterworth bandpass, constructs analytic signal via Hilbert transform: $Z(t) = x(t) + j\mathcal{H}\{x(t)\}$.
   - Computes instantaneous envelope amplitude $\mathcal{A}_{\text{THM}}(t)$ and instantaneous arterial compliance $\frac{d(\text{ePWV})}{dP} = \frac{\Delta \text{ePWV}}{\Delta P_{\text{hydro}}}$.
5. Implement `stiffness_solver.py`:
   - Closed-form analytical identification of arterial stiffness parameters using Moens-Korteweg and Hughes non-linear elasticity:
     $$\alpha = \frac{2 \cdot \ln(\text{ePWV}_2 / \text{ePWV}_1)}{\rho_{\text{blood}} \cdot g \cdot \Delta h}$$
     $$E_0 = \frac{2 \rho_{\text{blood}} R_0}{h_w} c_0^2, \quad c_0 = \text{ePWV}_{\text{collapse}}$$
     $$P_{\text{MAP}} = \frac{2}{\alpha} \ln\left( \frac{\text{ePWV}_1}{c_0} \right)$$
   - Decomposes into SBP and DBP via contact pulse morphology area factor $k_{\text{form}} \approx 0.33 - 0.40$ and $\text{PP}_{\text{HRVP}}$.
   - Exports the 6-dimensional physiological anchor vector $z_{\text{HRVP}} = [\text{ePWV}, \Delta PTT, \Delta P_{\text{hydro}}, \frac{d(\text{ePWV})}{dP}, E_0, \alpha]$.
6. Write comprehensive unit tests in `tests/unit/test_pinn.py` verifying sub-frame foot detection accuracy against injected delays, closed-form stiffness parameter recovery (< 1% error), and hydrostatic shift calculations.
7. Run the tests via `pytest tests/unit/test_pinn.py`, verify all pass, document results in `handoff.md`.

## 2026-09-23T18:16:31Z
You are worker_m2 (HRVP-PINN Hemodynamic Engine Specialist).
Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m2
Read C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md, C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md, and C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m2\DISPATCH.md.
Review Explorer 3 handoff in .agents/explorer_survey_3/handoff.md.
MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work.
Implement src/pinn/ (foot_detector.py, anthropometrics.py, hydrostatic_engine.py, respiratory_envelope.py, stiffness_solver.py) per specifications.
Write comprehensive unit tests in tests/unit/test_pinn.py.
Run pytest tests/unit/test_pinn.py, verify passing, document results in handoff.md. Notify orchestrator via send_message.
