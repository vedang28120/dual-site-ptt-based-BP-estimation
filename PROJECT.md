# Project: Dual-Site Blood Pressure Estimation Framework

## Architecture
The Dual-Site Blood Pressure Estimation Framework achieves calibration-free, cuffless blood pressure monitoring on mobile edge hardware (Xiaomi Poco M6 Pro 5G / Qualcomm Snapdragon 4 Gen 2) by coupling synchronized dual-site optical signals with a Physics-Informed Hydrostatic-Respiratory Vasomotor Perturbation (HRVP-PINN) engine and an Anti-Collapse Dual-Stream Temporal Convolutional Network (DS-TCN).

```
   [Front Camera (Face)]               [Rear Camera (Finger)]               [6-Axis IMU (Tilt)]
            │                                   │                                    │
   (Direct YUV Sampling)               (Direct 64x64 Crop)                 (AHRS Gravity + ZUPT)
   (Decoupled MediaPipe)               (Torch + Manual AE)                           │
            │                                   │                                    │
            ▼                                   ▼                                    ▼
   Front RGB Stream                    Rear RGB Stream                      Elevation Displacement
   (t_front_ns, Face)                  (t_rear_ns, Finger)                  (delta h, delta P_hydro)
            │                                   │                                    │
            └─────────────────┬─────────────────┘                                    │
                              ▼                                                      │
               Master Time-Grid PCHIP Resampling (125 Hz)                            │
               (Eliminates thread jitter & rolling-shutter phase)                    │
                              │                                                      │
            ┌─────────────────┴─────────────────┐                                    │
            ▼                                   ▼                                    │
   Facial rPPG Waveform                Finger Contact Waveform                       │
   (Macro-Foot Timing)                 (High-SNR Morphology)                         │
            │                                   │                                    │
            └─────────────────┬─────────────────┘                                    │
                              ▼                                                      │
             Sub-Frame Intersecting Tangent Foot Detector                            │
             (Spline continuous roots, <= 0.75 ms precision)                         │
                              │                                                      │
                              ▼                                                      │
                  Delta PTT -> Baseline ePWV                                         │
                              │                                                      │
                              └─────────────────┬────────────────────────────────────┘
                                                ▼
                                    HRVP-PINN Hemodynamic Engine
                        (Traube-Hering-Mayer 0.2Hz Hilbert envelope d(ePWV)/dP)
                        (Closed-form identification of arterial stiffness E_0, alpha)
                        (Calculates baseline calibration-free P_MAP, SBP, DBP)
                                                │
                                                ▼ Physiological Anchor z_HRVP
                                   +-------------------------+
                                   | Anti-Collapse DS-TCN    |
                                   | - Stream A: Finger PPG  |
                                   | - Stream B: Face rPPG   |
                                   | - Causal Cross-Attention|
                                   | - Decoupled SBP/DBP     |
                                   | - INT8 Quantized        |
                                   +------------┬------------+
                                                │
                                                ▼
                                   Calibrated Continuous Blood Pressure
                                   (SBP, DBP, MAP, Instantaneous ePWV)
```

## Feature Inventory
Every feature identified during the survey phase is mapped to its designated milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `dual_camera_capture` | Android Camera2 concurrent dual-stream capture (480p @ 30 FPS across front/rear) | M1 | Survey (R1) |
| 2 | `rear_anti_saturation` | Rear contact PPG with manual exposure (ISO 50, 1.0 ms) and direct 64x64 YUV center crop | M1 | Survey (R1) |
| 3 | `front_decoupled_tracking`| Front face ROI tracking via decoupled dual-rate MediaPipe face mesh (zero-copy fast path) | M1 | Survey (R1) |
| 4 | `nanosecond_pchip_sync` | Nanosecond SENSOR_TIMESTAMP binding + Master Time-Grid PCHIP Resampling (125 Hz) | M1 | Survey (R1) |
| 5 | `subframe_tangent_foot` | Sub-frame continuous intersecting tangent spline foot detector (<= 0.75 ms precision) | M2 | Survey (R2) |
| 6 | `epwv_path_calculator` | Anthropometric path length $D_{\text{path}} = 0.55 \cdot \text{Height} + 0.12 \cdot \text{ArmSpan}$ & ePWV | M2 | Survey (R2) |
| 7 | `imu_hydrostatic_engine` | IMU two-elevation hydrostatic perturbation engine ($\Delta P_{\text{hydro}} = \rho g \Delta h$) with ZUPT | M2 | Survey (R2) |
| 8 | `thm_envelope_compliance`| Traube-Hering-Mayer 0.2 Hz respiratory envelope extraction via Hilbert transform & $d(\text{ePWV})/dP$ | M2 | Survey (R2) |
| 9 | `arterial_stiffness_solver`| Closed-form identification of arterial stiffness parameters ($E_0, \alpha$) & baseline MAP | M2 | Survey (R2) |
| 10 | `ds_tcn_architecture` | Dual-Stream Dilated Causal TCN with causal cross-attention kernel (~190k params) | M3 | Survey (R3) |
| 11 | `hrvp_anchor_fusion` | Physiological anchor embedding ($z_{\text{HRVP}} \in \mathbb{R}^6$) & decoupled SBP/DBP heads | M3 | Survey (R3) |
| 12 | `tripartite_loss_trainer`| Tripartite loss optimizer ($\mathcal{L}_{\text{Huber}} + 0.40(1 - \text{Pearson}_r) + 0.25\mathcal{L}_{\text{Variance}}$) | M3 | Survey (R3) |
| 13 | `int8_quantization` | INT8 quantization via TFLite/ONNX (< 60 ms Snapdragon 4 Gen 2 runtime, <= 20 MB RAM) | M3 | Survey (R3) |
| 14 | `mobile_system_integration`| End-to-end mobile inference pipeline bridging optical engine, PINN solver, and INT8 model | M4 | Survey (R1-R3) |
| 15 | `template_collapse_audit` | Programmatic verification & Diagnostic Template Collapse Audit ($r \ge 0.82$, AAMI/BHS Grade A) | Final | Survey (Verification) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Independent Opaque-Box Test Suite (Tiers 1-4), Synthetic physiological generators, MIMIC/MCD-Iriun validation harness, Template Collapse Auditor, publish `TEST_READY.md` | none | PLANNED |
| M1 | Synchronized Optical Engine | Android Camera2 dual-stream, manual anti-saturation, 64x64 center crop, decoupled MediaPipe, SENSOR_TIMESTAMP PCHIP synchronizer | none | PLANNED |
| M2 | Physics-Informed Hemodynamic Engine (HRVP-PINN) | Sub-frame intersecting tangent foot detector, D_path & ePWV, IMU hydrostatic tilt, THM respiratory envelope compliance, closed-form E0/alpha solver | none | PLANNED |
| M3 | Anti-Collapse Edge DL Architecture (DS-TCN) | Dual-stream dilated causal TCN, causal cross-attention, HRVP anchor MLP, tripartite compound loss, INT8 quantization pipeline | M2 | PLANNED |
| M4 | Mobile System Integration | End-to-end integration of Optical Engine, HRVP-PINN, and INT8 TFLite runtime on Snapdragon 4 Gen 2 profile | M1, M2, M3 | PLANNED |
| Final | Final Milestone & Template Collapse Hardening | Phase 1: 100% pass on E2E test suite (Tiers 1-4); Phase 2: Tier 5 Adversarial Coverage Hardening via Challengers + Forensic Integrity Audit | E2E, M4 | PLANNED |

## Interface Contracts

### 1. Optical Engine ↔ Synchronizer Contract
- **Input**: Front camera `Image` (YUV_420_888) with `SENSOR_TIMESTAMP`, Rear camera `Image` (YUV_420_888) with `SENSOR_TIMESTAMP`.
- **Output**: Synchronized 125 Hz uniform signals:
  - `master_time_sec`: `np.ndarray` of shape `(T,)`
  - `face_rppg`: `np.ndarray` of shape `(T, 3)` (Forehead/Cheek spatial RGB)
  - `finger_ppg`: `np.ndarray` of shape `(T, 3)` (64x64 center crop spatial RGB)
- **Data Types**: `float32`, nanosecond timestamps `int64`.

### 2. Synchronizer ↔ HRVP-PINN Engine Contract
- **Input**:
  - `face_signal`: `np.ndarray` shape `(T,)` (filtered POS or green channel)
  - `finger_signal`: `np.ndarray` shape `(T,)` (filtered contact green channel)
  - `imu_data`: `dict` with keys `{"delta_h_m": float, "hold_stable": bool}`
  - `user_profile`: `dict` with keys `{"height_m": float, "arm_span_m": Optional[float]}`
- **Output**:
  - `hrvp_output`: `dict` containing:
    - `"delta_ptt_ms"`: `float` (sub-frame transit delay)
    - `"epwv_mps"`: `float` (estimated pulse wave velocity)
    - `"alpha"`: `float` (arterial stiffness pressure-dependence coefficient)
    - `"e0_pa"`: `float` (zero-pressure elastic modulus)
    - `"compliance_slope"`: `float` ($d(\text{ePWV})/dP$)
    - `"map_analytical_mmhg"`: `float`
    - `"sbp_analytical_mmhg"`: `float`
    - `"dbp_analytical_mmhg"`: `float`
    - `"anchor_vector"`: `np.ndarray` shape `(6,)` (`[epwv, delta_ptt, delta_p_hydro, compliance, e0, alpha]`)

### 3. HRVP-PINN ↔ DS-TCN Neural Model Contract
- **Input**:
  - `morph_input`: `torch.Tensor` or `np.ndarray` shape `(B, 1, 300)` (contact finger PPG normalized)
  - `timing_input`: `torch.Tensor` or `np.ndarray` shape `(B, 1, 300)` (facial rPPG normalized)
  - `anchor_input`: `torch.Tensor` or `np.ndarray` shape `(B, 6)` (standardized HRVP physiological anchor)
- **Output**:
  - `sbp_pred`: `Tensor` shape `(B, 1)`
  - `dbp_pred`: `Tensor` shape `(B, 1)`
- **INT8 Format**: TFLite flatbuffer (`dual_stream_tcn_int8.tflite`) with input quantization scales.

## Code Layout
```
C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\
├── .agents/                          # Agent metadata (BRIEFING, DISPATCH, handoffs)
├── PROJECT.md                        # Master project index & contracts
├── TEST_INFRA.md                     # E2E Test infrastructure & coverage index
├── TEST_READY.md                     # Test readiness signal (created by Test Track)
├── src/
│   ├── optical/                      # Optical Engine (Requirement R1)
│   │   ├── __init__.py
│   │   ├── camera_config.py          # Android Camera2 manual exposure & session settings
│   │   ├── crop_extractor.py         # 64x64 center crop direct YUV extraction
│   │   ├── face_roi_tracker.py       # Decoupled fast-path face ROI extraction
│   │   └── time_grid_synchronizer.py # Nanosecond SENSOR_TIMESTAMP PCHIP resampler
│   ├── pinn/                         # Physics-Informed Engine (Requirement R2)
│   │   ├── __init__.py
│   │   ├── foot_detector.py          # Sub-frame continuous intersecting tangent foot detector
│   │   ├── anthropometrics.py        # D_path & ePWV calculation
│   │   ├── hydrostatic_engine.py     # IMU hydrostatic perturbation (delta P = rho*g*delta h)
│   │   ├── respiratory_envelope.py   # Traube-Hering-Mayer 0.2Hz Hilbert envelope & d(ePWV)/dP
│   │   └── stiffness_solver.py       # Closed-form E0, alpha, and MAP solver
│   ├── models/                       # Edge Deep Learning Model (Requirement R3)
│   │   ├── __init__.py
│   │   ├── ds_tcn.py                 # Dual-Stream Dilated Causal TCN + Causal Cross-Attention
│   │   ├── compound_loss.py          # Tripartite loss (Huber + Pearson_r + Variance)
│   │   ├── dataset.py                # Dataset loader & batch generator
│   │   ├── trainer.py                # Training & validation loop
│   │   └── quantizer.py              # INT8 QAT export & TFLite/ONNX converter
│   └── mobile/                       # Mobile Integration (Requirement R4)
│       ├── __init__.py
│       ├── pipeline.py               # End-to-end inference orchestrator
│       └── tflite_runner.py          # Lightweight TFLite INT8 inference wrapper
├── tests/
│   ├── e2e/                          # Opaque-box E2E test suite (Tiers 1-4)
│   │   ├── test_tier1_feature.py
│   │   ├── test_tier2_boundary.py
│   │   ├── test_tier3_pairwise.py
│   │   └── test_tier4_workloads.py
│   ├── unit/                         # Unit tests for individual components
│   │   ├── test_optical.py
│   │   ├── test_pinn.py
│   │   └── test_models.py
│   └── audit/                        # Template Collapse & Forensic Integrity Audit
│       └── template_collapse_audit.py
└── data/                             # Synthetic & cached evaluation datasets
```
