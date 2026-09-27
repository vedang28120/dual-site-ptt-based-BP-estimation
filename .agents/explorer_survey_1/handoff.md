# Comprehensive Handoff Report — Reference Codebase & Baseline Survey

**Author**: Explorer 1 (Reference Codebase & Baseline Surveyor)  
**Target Recipient**: Orchestrator & Engineering Team  
**Date/Timestamp**: 2026-09-23T18:15:00Z  
**Working Directory**: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_1`  
**Reference Directory (Read-Only)**: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation`  
**Authoritative Blueprint**: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`

---

## 1. Observations

### 1.1 Repository Architecture & Module Inventory
The reference codebase `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation` encompasses 21 subdirectories and 44 root-level files, spanning research scripts, deep learning models, datasets, Android codebases, diagnostic benchmarks, and presentation materials:

1. **`MASTER_PROJECT_CONTEXT.md`** (Lines 1–541): The definitive technical reference for the previous "LuminaBP" mobile rPPG project.
2. **`core_extraction/`**:
   - `pos_extractor.py` (Lines 1–51): Implementation of Plane-Orthogonal-to-Skin (Wang et al., IEEE TBME 2016).
   - `chrom_extractor.py` (Lines 1–41): Implementation of Chrominance-based rPPG (de Haan & Jeanne, IEEE TBME 2013).
   - `green_extractor.py` (Lines 1–25): Baseline single green-channel intensity extraction.
   - `face_mesh_tracker.py` (Lines 1–99): MediaPipe 468-point face mesh tracking across 5 anatomical ROIs (Forehead, Left Cheek, Right Cheek, Nose, Chin) with 10 sub-patches.
   - `multi_patch_cca.py` (Lines 1–112): Canonical Correlation Analysis across left/right facial sub-patches and FastICA Blind Source Separation.
   - `liveness_detector.py` (Lines 1–55): Eye Aspect Ratio (EAR) blink detection and nose tip 3D depth variance.
3. **`filtering/`**:
   - `temporal_resampler.py` (Lines 1–39): Resampling VFR timestamps to uniform 125 Hz via `PchipInterpolator` or `CubicSpline`.
   - `dual_stream_filter.py` (Lines 1–37): 4th-order zero-phase Butterworth bandpass filter (`scipy.signal.filtfilt`, [0.75, 3.0] Hz) for heart rate.
   - `wavelet_denoiser.py` (Lines 1–58): DWT BayesShrink adaptive soft-thresholding using `sym8` wavelet.
   - `detrending.py` (Lines 1–33): Smoothness Priors Approach (SPA) detrending with regularization parameter $\lambda = 100$.
   - `sqi_validator.py` (Lines 1–62): Welch spectral SNR ($>2.0$ dB), skewness ($|\text{skew}| < 2.0$), and kurtosis ($<5.0$) quality gates.
   - `topological_mai.py` (Lines 1–155): Melanin Absorption Invariance (MAI) framework with Takens' delay embedding ($m=3, \tau=2$), phase-space attractor metrics ($f_1$ to $f_5$), and Recurrence Quantification Analysis (RQA).
4. **`models/`**:
   - `models/architectures/resnet_bigru_attn.py` (Lines 1–194): MODEL-06: Dual-Branch 1D-ResNet (Branch 1: $k=5$; Branch 2: $k=11$, dilated) + 2-layer BiGRU ($h=64$) + 4-head Multi-Head Self-Attention (MHSA) + Demographic MLP fusion + decoupled SBP/DBP heads.
   - `models/architectures/drp_bbp_net.py` (Lines 1–108): BBP-Net with 6-channel input tensor $[y_f, y_a, y'_f, y'_a, y''_f, y''_a]$ and `ScaledSigmoidBP` physiological bounding (SBP: [80, 180], DBP: [60, 130] mmHg).
   - `models/architectures/bayesian_ufacebp.py` (Lines 1–210): Bayesian U-FaceBP architecture with Monte Carlo dropout and heteroscedastic uncertainty heads ($\mu, \log \sigma^2$).
   - `models/architectures/legacy_lstm.py` (Lines 1–45): Multi-layer LSTM ingesting $(1, 875, 3)$ tensor.
   - `models/architectures/demo_mlp.py` (Lines 1–30): MODEL-01 baseline regressing demographics `[age, sex, bmi]` to BP.
   - `models/training/train_bp_mcd.py` (Lines 1–1127): Progressive ablation trainer (Phases 1–6) downloading `wengziheng/mcd_rppg` from Hugging Face.
   - `models/checkpoints/`: Frozen checkpoints: `MODEL-06-SepHead_original.pth` (3.3 MB), `MODEL-06-SepHead_balanced_moderate.pth` (3.3 MB), `MODEL-06-SepHead_balanced_strong.pth` (3.3 MB), and `lstm_ppg_nonmixed.h5` (2.5 MB).
5. **`PTT_BP_Estimation/`**:
   - `PTT_BP_Estimation/README.md` (Lines 1–45): Defines separate timing-based PTT/PAT research pipeline. Strictly rejects `facial_roi_pair` and `single_rppg` configurations due to shared camera clock and sub-millimeter proximity.
   - `ptt_bp/timing.py` (Lines 1–47): PAT/PTT extraction using `scipy.signal.find_peaks` and naive discrete `np.argmin` foot detection.
   - `ptt_bp/synchronization.py` (Lines 1–30): Resamples signals to common interval via `PchipInterpolator`.
   - `ptt_bp/estimation.py` (Lines 1–40): RandomForestRegressor multi-output model on timing features.
   - `ptt_bp/features.py` (Lines 1–28): Calculates `ptt_median_ms`, `ptt_std_ms`, `inverse_ptt_per_s`, and `heart_rate_bpm`.
6. **`mobile/` and `camera pipeline/` (Android App)**:
   - `mobile/android_app/app/src/main/java/com/rppg/bpestimation/CameraService.kt` (Lines 1–257): Android Camera2 implementation with AE/AWB lock, 640x480 resolution target, `ImageReader` in `YUV_420_888` format. Only operates a single camera at a time (switches via `switchCamera()`).
   - `mobile/android_app/app/src/main/java/com/rppg/bpestimation/FaceLandmarkTracker.kt` (Lines 1–339): MediaPipe Face Landmarker. Extracts RGB using fast EMA bounding boxes on YUV planes, but slow path converts YUV -> NV21 -> JPEG compression (`YuvImage.compressToJpeg`) -> Bitmap decoding (`BitmapFactory.decodeByteArray`) on CPU.
   - `mobile/android_app/app/src/main/java/com/rppg/bpestimation/BPInferenceEngine.kt` (Lines 1–88): Loads `lstm_ppg_nonmixed.tflite` (851 KB) expecting `[1, 875, 3]` input tensor. Forces `sbp = maxOf(val1, val2)` and `dbp = minOf(val1, val2)`.
   - `mobile/android_app/app/src/main/python/pos_engine.py` (Lines 1–364): Chaquopy Python DSP backend executing Cubic Spline interpolation, POS extraction, Butterworth bandpass, DWT BayesShrink, SPA detrending, Savitzky-Golay smoothing, Welch PSD HR, and Pan-Tompkins HRV.

---

### 1.2 Verbatim Evidence of Template Collapse (Regression to the Mean)

#### Evidence Item A: Method-Invariant Static Predictions (`results/mcd_1020_accuracy_comparison.json`)
When benchmarking four fundamentally different algorithms (Deep Learning TS-CAN, Physics-based POS, Chrominance CHROM, and Green channel) on subject 1020 across both resting and post-exercise states:

```json
"before": {
  "meta": { "gt_sbp": 105.0, "gt_dbp": 78.0, "state": "Resting / Baseline" },
  "methods": {
    "TS-CAN": { "sbp": 127.0, "dbp": 76.2 },
    "POS":     { "sbp": 127.1, "dbp": 76.3 },
    "CHROM":   { "sbp": 127.2, "dbp": 76.4 },
    "GREEN":   { "sbp": 127.7, "dbp": 76.7 }
  }
},
"after": {
  "meta": { "gt_sbp": 113.0, "gt_dbp": 78.0, "state": "Post-Exercise / Elevated HR & SBP" },
  "methods": {
    "TS-CAN": { "sbp": 127.0, "dbp": 76.2 },
    "POS":     { "sbp": 127.1, "dbp": 76.3 },
    "CHROM":   { "sbp": 127.2, "dbp": 76.4 },
    "GREEN":   { "sbp": 127.8, "dbp": 76.8 }
  }
}
```
*Direct observation*: Ground truth SBP shifted by $+8.0\text{ mmHg}$ post-exercise ($105.0 \to 113.0\text{ mmHg}$). Yet across all four algorithms, predicted SBP remained completely frozen at $127.0 - 127.8\text{ mmHg}$ (variance $< 0.1\text{ mmHg}$), and predicted DBP remained frozen at $76.2 - 76.8\text{ mmHg}$.

#### Evidence Item B: Negative and Zero Pearson Correlations (`camera pipeline/tests/accuracy_report_50subjects.md`)
On a 50-subject benchmark of the MCD dataset:
- **Systolic BP (SBP)**: MAE $= 16.85\text{ mmHg}$, RMSE $= 20.50\text{ mmHg}$, Pearson $r = \mathbf{-0.311}$
- **Diastolic BP (DBP)**: MAE $= 7.87\text{ mmHg}$, RMSE $= 10.23\text{ mmHg}$, Pearson $r = \mathbf{-0.027}$
- **Heart Rate (HR)**: MAE $= 33.84\text{ BPM}$

Notable subject samples:
- Subject 1149 (Actual SBP $152.0\text{ mmHg}$): Predicted SBP $= \mathbf{104.5\text{ mmHg}}$ (Error $= 47.5\text{ mmHg}$)
- Subject 1113 (Actual SBP $95.0\text{ mmHg}$): Predicted SBP $= \mathbf{118.1\text{ mmHg}}$ (Error $= 23.1\text{ mmHg}$)
- Subject 1292 (Actual SBP $152.0\text{ mmHg}$): Predicted SBP $= \mathbf{111.5\text{ mmHg}}$ (Error $= 40.5\text{ mmHg}$)
*Direct observation*: The model exhibits an inverted slope; hypertensive patients are severely underpredicted while hypotensive patients are overpredicted, clustering into the normotensive band.

#### Evidence Item C: Compression of Prediction Variance (`camera pipeline/tests/gold_pipeline_ablation_report.md`, Lines 50–57)
In the Phase 6 Regression-to-Mean Diagnostic:
- True SBP Range: $85.0\text{ mmHg}$ ($80 - 165\text{ mmHg}$)
- Predicted SBP Range: $31.4\text{ mmHg}$ ($105 - 136\text{ mmHg}$) $\implies$ Spread Ratio $\frac{\sigma_{\text{pred}}}{\sigma_{\text{true}}} = \mathbf{0.37}$
- True DBP Range: $49.0\text{ mmHg}$ ($50 - 99\text{ mmHg}$)
- Predicted DBP Range: $18.0\text{ mmHg}$ ($64 - 82\text{ mmHg}$) $\implies$ Spread Ratio $\frac{\sigma_{\text{pred}}}{\sigma_{\text{true}}} = \mathbf{0.37}$
Line 56 records:
> `[WARNING]: SBP predictions occupy <50% of true range -- likely regression-to-mean!`

#### Evidence Item D: Failure on Direct Contact PPG Sensors (`camera pipeline/tests/accuracy_report_lstm_direct_ppg.md`, Lines 7–11)
When evaluated directly on clinical contact PPG sensors (`.PW` files) without any camera or rPPG extraction:
- SBP MAE $= 20.09\text{ mmHg}$, RMSE $= 24.00\text{ mmHg}$, Pearson $r = \mathbf{-0.205}$
- DBP MAE $= 13.52\text{ mmHg}$, RMSE $= 15.71\text{ mmHg}$, Pearson $r = \mathbf{-0.080}$
*Direct observation*: Template collapse is not merely an optical extraction artifact; sequence models trained on single-site morphology alone fail to learn genuine hemodynamic variation.

#### Evidence Item E: Static Model Drift in Balancing Experiments (`results/metrics/balancing_comparison.csv`)
```csv
experiment,win_sbp_mae,win_sbp_rmse,win_sbp_r,win_dbp_mae,win_dbp_rmse,win_dbp_r,sub_sbp_mae,sub_dbp_mae
EXP-A (Original),110.82338,110.868065,-0.6882425225217557,77.19414,77.19414,0.0,110.82338,77.19414
EXP-B (Moderate),111.074844,111.11914,0.15158585057253457,77.5708,77.57081,0.0,111.074844,77.57081
EXP-C (Strong),110.89993,110.94455,-0.7653184972776759,77.189156,77.189156,0.0,110.899925,77.189156
```
*Direct observation*: Across all balancing regimes (Original, Moderate, Strong), `win_dbp_r` is identically $0.000$, and MAE equals RMSE to three decimal places ($77.194$ vs $77.194$), indicating that the model's DBP prediction is an unyielding constant.

---

### 1.3 Dataset Architectures & Ground Truth Formats

| Dataset | Storage Path / Source | Format & Files | Sampling Rate | Annotations / Ground Truth | Alignment / Protocol |
|:---|:---|:---|:---|:---|:---|
| **MCD-Iriun (Multi-Camera Dataset)** | Hugging Face: `wengziheng/mcd_rppg` | `db.csv` metadata;<br>`ppg/{pid}_{step}.PW` (ASCII floating point);<br>`ppg_sync/{pid}_IriunWebcam_{step}.txt` or `.avi` | PPG: $125\text{ Hz}$<br>Video: $30\text{ FPS}$ | `upper_ap` (SBP, mmHg),<br>`lower_ap` (DBP, mmHg),<br>`age`, `sex`, `bmi`, `weight`, `height` | Step denotes condition (`before` = resting, `after` = post-exercise). Clean subject splits: 419 Train / 90 Val / 90 Test. |
| **MIMIC-III Clinical Database** | Used for `lstm_ppg_nonmixed.h5` | Contact finger PPG + radial arterial catheter A-line | $125\text{ Hz}$ | Continuous beat-by-beat arterial blood pressure (invasive ground truth) | Pretraining only. Large domain gap when transferred to camera rPPG. |
| **Local Traces** | `data/raw/rgb_trace_*.csv` | CSV traces with 15 spatial columns (`CF_R, CF_G, CF_B, LF_R, ...`) | $30\text{ FPS}$ (Nominal VFR) | **No hardware timestamps**, no cuff ground truth | Cannot be used for PTT training or clinical validation. |
| **PTT Manifest Format** | `PTT_BP_Estimation/config/example_config.json` | JSON records mapping `signals.ecg`, `signals.ppg` or `proximal_ppg`, `distal_ppg` | Variable (resampled to $125\text{ Hz}$) | `sbp`, `dbp` cuff labels | Requires declaration of `{"method": "shared_hardware_clock", "validated": true}`. |

---

## 2. Logic Chain

The experimental and codebase findings construct an unbroken causal chain explaining why the reference system collapsed and what the new Dual-Site architecture must do:

### Step 1: Microvascular Attenuation & Quantization Degradation (Physical Barrier)
- *Observation*: `MASTER_PROJECT_CONTEXT.md` (Lines 438–442) and `fig_windkessel_damping.png`.
- *Deduction*: In human arterial hemodynamics, the dicrotic notch encodes peripheral resistance and diastolic pressure. As the pressure pulse travels through the Windkessel tree to facial microvasculature, high-frequency components are mechanically damped by 1–2 orders of magnitude. 
- *Consequence*: The 8-bit quantization floor of mobile sensors (256 discrete levels $\approx 0.39\%$ step size) buries the tiny $0.1 - 2.0\%$ pulsatile rPPG AC signal. The morphological dicrotic notch falls below the camera's physical quantization floor.

### Step 2: Temporal Undersampling at 30 FPS (Nyquist Barrier)
- *Observation*: `MASTER_PROJECT_CONTEXT.md` (Lines 442–443) and `CameraService.kt` (Line 175).
- *Deduction*: A 30–60 ms dicrotic notch captured at 30 FPS ($33.3\text{ ms}$ sample interval) spans at most 1–2 raw discrete samples.
- *Consequence*: Resampling to 125 Hz via PCHIP or Cubic Spline (`temporal_resampler.py`) smooths these 2 points into 8 points, but generates zero new physiological information. The true Nyquist limit remains 15 Hz.

### Step 3: Filter-Induced Destruction of Physiological Modulation (DSP Defect)
- *Observation*: `detrending.py` (Lines 10–32) and `pos_engine.py` (Line 254).
- *Deduction*: The Smoothness Priors Approach with $\lambda = 100$ applies a high-pass cutoff around $0.35 - 0.4\text{ Hz}$. 
- *Consequence*: This removes low-frequency baseline drift, but simultaneously eradicates the natural $0.15 - 0.3\text{ Hz}$ respiratory blood pressure modulation (Traube-Hering-Mayer waves) and vasomotor fluctuations. This strips the exact physiological perturbation needed to determine vascular compliance $\frac{d(\text{ePWV})}{dP}$.

### Step 4: Indeterminate Morphological Mapping & Loss Minimization (Mathematical Cause)
- *Observation*: `models/training/train_bp_mcd.py` (Lines 265–279) and `gold_pipeline_ablation_report.md` (Lines 23–32).
- *Deduction*: The neural network was trained using MSE/Huber loss on single-site waveforms against static cuff labels.
- *Consequence*: When input waveforms are dominated by sensor noise and lack distinct morphological features, gradient descent finds the global minimum of the loss function by setting the network weights to predict the statistical expectation of the training distribution:
  $$\hat{y} \to E[y] \approx 120 / 72\text{ mmHg}$$
  This minimizes the global error over a bell-shaped normotensive dataset, but produces near-zero or negative correlation ($r = -0.311$) across subjects. MODEL-01 (demographics only) achieved SBP MAE $= 11.85$, while the 12-layer ResNet-BiGRU-MHSA achieved $11.58$ — proving that the deep network was essentially functioning as a demographic lookup table.

### Step 5: Input Unnormalization in Production Serving Engine (Code Bug)
- *Observation*: `prototype_tscan_bp/bp_estimator.py` (Lines 108–110) vs `train_bp_mcd.py` (Lines 321–324).
- *Deduction*: During training, demographics were normalized:
  $$\text{age}_{\text{norm}} = \frac{\text{age} - 28.4}{10.2}, \quad \text{bmi}_{\text{norm}} = \frac{\text{bmi} - 22.1}{3.5}$$
  In `bp_estimator.py`, raw unnormalized values (`age = 23.0`, `bmi = 19.03`) were passed into the demographic MLP with weights trained for standardized inputs.
- *Consequence*: The demographic MLP layers saturated into extreme nonlinear regimes, completely drowning out the wave tensor features and pinning the output to a static point ($127.1 / 76.3\text{ mmHg}$) regardless of input signal changes.

### Step 6: Single-Camera Architecture & JNI Latency Bottlenecks (Hardware/OS Limit)
- *Observation*: `CameraService.kt` (Lines 35–44, 87–109) and `FaceLandmarkTracker.kt` (Lines 281–337).
- *Deduction*: The existing Android app operates only a single camera stream. When MediaPipe face tracking runs on CPU via YUV -> NV21 -> JPEG compression -> Bitmap decoding, processing exceeds 33 ms per frame.
- *Consequence*: The queue drops incoming frames (`isAnalyzing.get()` flag), introducing massive variable frame jitter. Furthermore, without hardware monotonic nanosecond timestamps (`SENSOR_TIMESTAMP`), inter-frame timing cannot achieve sub-millisecond precision.

---

## 3. Caveats

1. **Gated Dataset Access**: The full MCD-Iriun dataset (`wengziheng/mcd_rppg`) requires a Hugging Face user access token. Local evaluations in the reference repository relied on cached subsets (e.g. 50-subject benchmark).
2. **Contact Sensor Benchmark vs Camera Benchmark**: The 50-subject direct contact PPG benchmark (`accuracy_report_lstm_direct_ppg.md`) proved single-site morphology failure on pure PPG waveforms, but was conducted on finger photoplethysmograms rather than simultaneous dual-site pulses.
3. **No Direct Hardware Access During Survey**: Device thermal and battery benchmarks on the Poco M6 Pro 5G Snapdragon 4 Gen 2 mobile testbed were surveyed via report documentation (`LuminaBP_Vocational_Training_Report.md`, `rppg_to_bp_master_roadmap-v2.md`) and log files (`logcat.txt`), rather than active runtime profiling on a connected physical handset.

---

## 4. Conclusions

1. **Single-Site Optical Morphology Cannot Solve Cuffless BP**: Attempting to estimate continuous blood pressure from facial rPPG alone (or finger PPG alone) is mathematically and physiologically ill-posed. Dicrotic notch attenuation, 8-bit quantization noise, and demographic dominance inevitably trigger Template Collapse.
2. **Dual-Site Pulse Transit Time ($\Delta PTT$) is the Essential Foundation**: Measuring the transit time between a proximal site (facial rPPG / carotid artery) and a distal site (finger contact PPG / digital artery) provides a direct, calibration-anchored physical measurement of Pulse Wave Velocity ($\text{PWV} = D / \Delta PTT$), which is governed by the Bramwell-Hill and Moens-Korteweg equations ($PWV = \sqrt{\frac{V \cdot \Delta P}{\rho \cdot \Delta V}}$).
3. **The Hydrostatic-Respiratory Vasomotor Perturbation (HRVP-PINN) Engine Resolves Calibration**:
   - By using the phone's IMU to guide a two-elevation hydrostatic maneuver ($\Delta P_{\text{hydro}} = \rho g \Delta h$), the system introduces a known physical pressure perturbation without an external cuff.
   - By preserving (rather than filtering out) the $\sim 0.2\text{ Hz}$ Traube-Hering-Mayer respiratory blood pressure modulation, the system can compute instantaneous arterial compliance $\frac{d(\text{ePWV})}{dP}$, resolving individual baseline arterial stiffness ($E_0, \alpha$) in a calibration-free manner.
4. **Android Dual-Stream Camera2 Pipeline Must Be Built from Scratch**:
   - Must run concurrent 480p @ 30 FPS streams across front (facial rPPG) and rear (finger contact PPG with flash LED).
   - Must bind frames using `CaptureResult.SENSOR_TIMESTAMP` (nanosecond hardware clocks) rather than OS system time to eliminate thread-scheduling jitter.
   - Must eliminate JNI image conversion overhead (zero JPEG/Bitmap conversions) and use lock-free ring buffers to achieve zero dropped frames over a 10-second acquisition window.
5. **Anti-Collapse Edge Deep Learning Architecture**:
   - The Dual-Stream Temporal Convolutional Network (DS-TCN) must ingest contact PPG morphology, facial macro-foot timing, and the HRVP physiological anchor.
   - The compound loss function $\mathcal{L} = \mathcal{L}_{\text{Huber}} + \lambda_1 (1 - \text{Pearson}_r) + \lambda_2 \mathcal{L}_{\text{Variance}}$ will explicitly penalize variance collapse ($\sigma^2(\hat{y}) / \sigma^2(y) < 0.8$) and negative correlation.
   - Quantization to INT8 flatbuffer via TFLite will ensure local execution $<60\text{ ms}$ on Snapdragon 4 Gen 2 with peak memory $\le 20\text{ MB}$.

---

## 5. Verification Method

To independently verify the survey findings and diagnostic observations documented in this report:

1. **Verify Template Collapse in Diagnostic Logs**:
   - Inspect `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation\results\mcd_1020_accuracy_comparison.json`. Confirm that lines 21–24 and lines 99–102 show identical SBP ($127.0 - 127.8$) and DBP ($76.2 - 76.8$) predictions across resting and exercise states.
   - Inspect `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation\camera pipeline\tests\accuracy_report_50subjects.md`. Verify lines 9–10 documenting SBP Pearson $r = -0.311$ and DBP Pearson $r = -0.027$.
   - Inspect `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation\camera pipeline\tests\accuracy_report_lstm_direct_ppg.md`. Verify lines 9–10 documenting direct PPG Pearson $r = -0.205$.
2. **Verify Demographic Saturation Bug**:
   - Inspect `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation\prototype_tscan_bp\bp_estimator.py` lines 108–110. Observe that raw `age` and `bmi` are fed to `self.model(wave_tensor, demo_tensor)` without cohort standardization (`age_mean=28.4`, `bmi_mean=22.1`).
3. **Verify Baseline Discrete Foot Detection Limitation**:
   - Inspect `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation\PTT_BP_Estimation\ptt_bp\timing.py` lines 16–24. Observe that `_ppg_feet` uses discrete integer index `np.argmin(ppg[left:peak+1])`, which at 30 FPS imposes a $33.3\text{ ms}$ quantization step.
4. **Verify Detrending Respiratory Frequency Removal**:
   - Inspect `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation\filtering\detrending.py` lines 10–32. Observe that SPA detrending with $\lambda=100$ penalizes second differences and attenuates signals below $0.4\text{ Hz}$, eradicating respiratory Traube-Hering-Mayer modulations.

---
*End of Comprehensive Handoff Report — Explorer 1*
