# E2E Test Infrastructure & Opaque-Box Quality Assurance Specification

> **Project**: Dual-Site Blood Pressure Estimation Framework  
> **Target Device**: Xiaomi Poco M6 Pro 5G (Qualcomm Snapdragon 4 Gen 2 / SM4450)  
> **Testing Architecture**: 4-Tier Opaque-Box E2E Suite + Forensic Template Collapse Audit  
> **Author**: `test_writer_1` (E2E Testing Architect & Suite Writer)  
> **Status**: ACTIVE

---

## 1. Executive Summary & Verification Strategy

The core failure mode of prior camera-based blood pressure estimation systems is **Template Collapse (Regression to the Mean)**, wherein neural networks minimize MSE loss by predicting constant population norms ($\hat{y} \approx 120/75$ mmHg) with collapsed variance ($\sigma^2(\hat{y}) \to 0$) and near-zero or negative Pearson correlation ($r \le 0$).

This test infrastructure implements an **authoritative, independent, opaque-box testing framework** designed to rigorously validate:
1. Low-overhead dual-camera optical capture and nanosecond timestamp synchronization (`SENSOR_TIMESTAMP` PCHIP alignment at 125 Hz).
2. Sub-frame intersecting tangent pulse foot detection achieving $\le 0.75$ ms timing precision.
3. Physics-Informed Hemodynamic Engine (HRVP-PINN): anthropometric path calculation, IMU hydrostatic perturbation ($\Delta P_{\text{hydro}} = \rho g \Delta h$), Traube-Hering-Mayer 0.2 Hz respiratory envelope extraction ($d(\text{ePWV})/dP$), and closed-form arterial stiffness identification ($E_0, \alpha$).
4. Anti-collapse Dual-Stream Temporal Convolutional Network (DS-TCN) with causal cross-attention and tripartite compound loss ($\mathcal{L}_{\text{Huber}} + \lambda_1(1-r) + \lambda_2 \mathcal{L}_{\text{Var}}$).
5. Mobile runtime execution feasibility: Snapdragon 4 Gen 2 latency $<60$ ms, peak RAM $\le 20$ MB, INT8 quantization fidelity.

---

## 2. 4-Tier Test Architecture & Matrix

The testing hierarchy follows a strict 4-Tier methodology covering >= 170 discrete test cases:

```
+-----------------------------------------------------------------------------------+
|                        TIER 4: REAL-WORLD CLINICAL & MOBILE                       |
|   - 10-second continuous multi-subject acquisitions (MIMIC-III / MCD-Iriun traces)|
|   - Exercise stress transitions (resting 105 mmHg -> post-exercise 145 mmHg)     |
|   - Dynamic hydrostatic elevation sweeps (+30 cm, -25 cm)                         |
|   - Target: >= 8 long-duration end-to-end workload runs                           |
+---------------------------------------------------------+-------------------------+
                                                          |
+---------------------------------------------------------+-------------------------+
|                  TIER 3: PAIRWISE CROSS-FEATURE COMBINATIONS                      |
|   - High Heart Rate (160 BPM) + Hydrostatic Perturbation (300 mm)                |
|   - Severe VFR Timestamp Jitter (10 ms) + Sub-Frame Tangent Spline Foot Detection |
|   - Severe Optical Attenuation (20 dB SNR) + Decoupled MediaPipe ROI Tracking     |
|   - Low Vascular Compliance + Extreme Stiff Arteries (Hughes alpha = 0.040)       |
|   - Target: >= 15 multi-variable stress pairs                                     |
+---------------------------------------------------------+-------------------------+
                                                          |
+---------------------------------------------------------+-------------------------+
|                    TIER 2: BOUNDARY & CORNER CASES (STRESS)                       |
|   - Extreme Bradycardia (35 BPM) / Extreme Tachycardia (190 BPM)                 |
|   - Hypertensive Crisis (210/130 mmHg) / Severe Hypotension (75/45 mmHg)          |
|   - Stature extremes: Micro-height (1.30 m) / Macro-height (2.10 m)              |
|   - Micro-elevation (10 mm) / Macro-elevation (600 mm) / Zero-tilt boundary       |
|   - ISO 50 floor / LED flash saturation boundary / Quantization floor             |
|   - Target: >= 5 corner cases per feature (>= 75 tests)                           |
+---------------------------------------------------------+-------------------------+
                                                          |
+---------------------------------------------------------+-------------------------+
|                    TIER 1: ISOLATED FEATURE COVERAGE (UNIT/E2E)                   |
|   - Feature 1: dual_camera_capture (480p @ 30 FPS dual-stream)                    |
|   - Feature 2: rear_anti_saturation (Manual AE, ISO 50, 1.0 ms clamp)             |
|   - Feature 3: front_decoupled_tracking (MediaPipe fast-path zero-copy ROI)       |
|   - Feature 4: nanosecond_pchip_sync (Master time-grid PCHIP 125 Hz resampler)    |
|   - Feature 5: subframe_tangent_foot (Continuous spline root foot <= 0.75 ms)     |
|   - Feature 6: epwv_path_calculator (D_path = 0.55*H + 0.12*W and ePWV)           |
|   - Feature 7: imu_hydrostatic_engine (Delta P = rho*g*Delta h & ZUPT)            |
|   - Feature 8: thm_envelope_compliance (0.2 Hz Hilbert envelope & d(ePWV)/dP)     |
|   - Feature 9: arterial_stiffness_solver (Closed-form E_0, alpha, P_MAP)          |
|   - Feature 10: ds_tcn_architecture (Dual-stream causal dilated TCN cross-attn)   |
|   - Feature 11: hrvp_anchor_fusion (6-dim physiological anchor embedding)         |
|   - Feature 12: tripartite_loss_trainer (Huber + Pearson + Variance loss)         |
|   - Feature 13: int8_quantization (QAT INT8 scales & tensor bounds)               |
|   - Feature 14: mobile_system_integration (Full pipeline latency & memory profile)|
|   - Feature 15: template_collapse_audit (r >= 0.82, variance ratio 0.70 - 1.30)   |
|   - Target: >= 5 tests per feature (>= 75 tests)                                  |
+-----------------------------------------------------------------------------------+
```

### Detailed Count Budget

| Tier | Focus Scope | Requirement | Allocated Test Cases |
|:---|:---|:---:|:---:|
| **Tier 1** | Feature Coverage (Features 1–15) | $\ge 5$ per feature | **75 test cases** |
| **Tier 2** | Boundary & Corner Cases | $\ge 5$ per feature | **75 test cases** |
| **Tier 3** | Pairwise Combinations | $\ge 15$ combinations | **16 test cases** |
| **Tier 4** | Real-world Clinical Workloads | $\ge 8$ workloads | **8 test cases** |
| **Audit** | Diagnostic Template Collapse | Comprehensive cohort audit | **1 standalone audit** |
| **TOTAL** | Complete E2E Suite | $\ge 170$ | **174 test cases** |

---

## 3. Synthetic Physiological Waveform Generator Architecture

To provide ground-truth verification without clinical patient risk or dependency on external servers, the suite incorporates an authoritative analytical physiological generator (`tests/synthetic_generator.py`).

### 3.1 Mathematical Ground Truth Equations
1. **Arterial Pressure Pulse Profile (Two-Gaussian + Exponential Decay)**:
   $$P_{\text{aortic}}(t) = P_{\text{diastolic}} + \Delta P_{\text{pulse}} \cdot \left[ A_1 e^{-\frac{(t - \mu_1)^2}{2\sigma_1^2}} + A_2 e^{-\frac{(t - \mu_2)^2}{2\sigma_2^2}} \right]$$
   - Primary peak: Systolic upstroke ($\mu_1 \approx 0.12 T_{\text{cardiac}}$)
   - Secondary peak: Dicrotic notch reflection ($\mu_2 \approx 0.35 T_{\text{cardiac}}$)

2. **Arterial Elasticity & Hughes Law**:
   $$E_{\text{inc}}(P) = E_0 e^{\alpha P}, \quad \text{PWV}(P) = c_0 e^{\frac{\alpha}{2} P}$$
   $$\text{with } c_0 = \sqrt{\frac{h_w E_0}{2 \rho_{\text{blood}} R_0}}$$

3. **Pulse Transit Time ($\Delta PTT$)**:
   $$\Delta PTT = \frac{D_{\text{path}}}{\text{PWV}(P)} = \frac{0.55 \cdot \text{Height} + 0.12 \cdot \text{ArmSpan}}{c_0 e^{\frac{\alpha}{2} P}}$$

4. **Traube-Hering-Mayer Respiratory Modulation**:
   $$P(t) = P_{\text{base}} + \Delta P_{\text{THM}} \cdot \sin(2\pi f_{\text{resp}} t), \quad f_{\text{resp}} \approx 0.20\text{ Hz}$$
   $$\text{ePWV}(t) = c_0 e^{\frac{\alpha}{2} P(t)}$$

5. **IMU Hydrostatic Perturbation**:
   $$\Delta P_{\text{hydro}}(t) = \rho_{\text{blood}} \cdot g \cdot \Delta h(t)$$
   $$\rho_{\text{blood}} = 1055\text{ kg/m}^3, \quad g = 9.80665\text{ m/s}^2 \implies 77.601\text{ mmHg/m}$$

6. **Distal vs Proximal Waveform Synthesis**:
   - **Proximal Face rPPG**: Filtered with Windkessel damping factor $\gamma_{\text{face}} = 0.35$ (suppressing dicrotic notch to simulate microvascular compliance), shifted by arrival time $t_{\text{face}} = D_{\text{carotid}} / \text{PWV}$.
   - **Distal Contact PPG**: High-SNR sharp dicrotic notch morphology, shifted by arrival time $t_{\text{finger}} = D_{\text{arm}} / \text{PWV} = t_{\text{face}} + \Delta PTT$.

7. **Sensor & Acquisition Artifacts**:
   - Monotonic nanosecond clock generation with Variable Frame Rate (VFR) jitter ($\sigma_{\text{jitter}} = 1.5 - 5.0$ ms).
   - Sensor rolling-shutter phase offset between front and rear cameras ($\Delta \phi \in [0, 33.3]$ ms).
   - Additive Gaussian white noise ($\text{SNR} = 15 - 35$ dB).
   - Capillary blanching baseline shift for contact sensor.

---

## 4. Diagnostic Template Collapse Audit Specification

Located at `tests/audit/template_collapse_audit.py`, this diagnostic performs forensic analysis across a synthetic heterogeneous validation cohort of $N \ge 100$ simulated subjects spanning:
- Normotensive ($110-125$ / $70-80$ mmHg)
- Pre-hypertensive ($125-139$ / $80-89$ mmHg)
- Stage 1 & Stage 2 Hypertensive ($140-190$ / $90-120$ mmHg)
- Hypotensive ($80-100$ / $50-65$ mmHg)
- Pediatric/Young & Elderly subjects ($E_0 \in [0.4, 2.0] \times 10^5$ Pa, $\alpha \in [0.015, 0.040]$ $\text{mmHg}^{-1}$)

### Pass / Fail Audit Criteria

| Metric | Target Standard | Collapse Mode Detected if Failed |
|:---|:---|:---|
| **Pearson Correlation ($r_{\text{SBP}}$)** | $\mathbf{r \ge 0.82}$ | Flatline prediction / demographic mean reversion |
| **Pearson Correlation ($r_{\text{DBP}}$)** | $\mathbf{r \ge 0.80}$ | Constant diastolic bias |
| **Prediction Variance Ratio** | $\mathbf{0.70 \le \frac{\sigma^2(\hat{y})}{\sigma^2(y)} \le 1.30}$ | **Variance Vanishing / Squeeze** (proves predictions span full clinical range) |
| **Mean Absolute Error (SBP)** | $\mathbf{\le 5.0\text{ mmHg}}$ | Systematic regression bias |
| **Mean Absolute Error (DBP)** | $\mathbf{\le 3.5\text{ mmHg}}$ | Systematic diastolic bias |
| **Error Standard Deviation (SD)** | $\mathbf{\le 7.0\text{ mmHg}}$ | High dispersion / inconsistent tracking |
| **AAMI / ISO 81060-2 Pass** | $\text{Mean Error} \le 5.0\text{ mmHg}, \text{SD} \le 8.0\text{ mmHg}$ | Standard medical device certification threshold |
| **BHS Grade Criteria** | Grade A ($\ge 60\% \le 5\text{ mmHg}, \ge 85\% \le 10\text{ mmHg}$) | Clinical grading failure |

---

## 5. Master Test Runner Execution Protocol

The master test runner `run_e2e_tests.py` orchestrates execution across all four tiers and the forensic audit:

```bash
# Run complete test suite (Tiers 1-4)
python run_e2e_tests.py --all

# Run individual tier
python run_e2e_tests.py --tier 1
python run_e2e_tests.py --tier 2
python run_e2e_tests.py --tier 3
python run_e2e_tests.py --tier 4

# Run Template Collapse Audit
python tests/audit/template_collapse_audit.py

# Run standard pytest
pytest tests/e2e/ -v
```

Output: Formatted terminal tables, JSON artifact generation (`test_results_summary.json`), and exit status code 0 on complete pass.
