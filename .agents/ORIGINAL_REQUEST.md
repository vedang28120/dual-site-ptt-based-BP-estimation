# Original User Request

## Initial Request — 2026-09-23T18:05:11Z

# Teamwork Project Prompt — Launched

> Status: Launched  
> Requested team: Full research & development team  
> Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation  
> Reference directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation  
> Integrity mode: development  

## Directives
- **Unconstrained Rigor**: Take all time and computational depth necessary. Execute full-throttle deep research, implementation, and rigorous programmatic verification.
- **Dedicated Clean Workspace**: All new modules, tests, models, and artifacts must be developed in `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`. Use `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation` strictly as a read-only reference for previous findings, architectures, and logs.

## Project Description
Develop and empirically validate an edge-deployable, calibration-free blood pressure estimation framework for the Poco M6 Pro 5G using synchronized dual-site optical signals (facial rPPG + finger contact PPG) coupled with a novel Physics-Informed Hydrostatic-Respiratory Vasomotor Perturbation (HRVP-PINN) engine to mathematically conquer template collapse.

## Requirements

### R1. Low-Overhead Synchronized Optical Engine for Poco M6 Pro 5G
Implement an Android Camera2 dual-stream capture pipeline operating at 480p @ 30 FPS across front and rear sensors. Rear camera must run with flash LED active and 64x64 center-cropped digital artery contact PPG; front camera must track facial ROIs via MediaPipe. Bind frame pairs using nanosecond hardware monotonic clocks (`SENSOR_TIMESTAMP`) to eliminate OS-level thread-scheduling jitter.

### R2. Physics-Informed Hemodynamic Engine (HRVP-PINN)
Develop an analytical parameter-identification engine that:
1. Converts raw $\Delta PTT$ into Estimated Pulse Wave Velocity ($\text{ePWV}$) using user anthropometrics ($D_{\text{path}} = 0.55 \cdot \text{Height} + 0.12 \cdot \text{ArmSpan}$).
2. Uses the phone's IMU to guide a two-elevation hydrostatic perturbation ($\Delta P_{\text{hydro}} = \rho g \Delta h$).
3. Extracts natural respiratory blood pressure modulation ($\sim 0.2\text{ Hz}$ Traube-Hering-Mayer envelope) to compute instantaneous compliance $\frac{d(\text{ePWV})}{dP}$, resolving individual baseline arterial stiffness ($E_0, \alpha$) without an external cuff.

### R3. Anti-Collapse Edge Deep Learning Architecture
Design a lightweight Dual-Stream Temporal Convolutional Network (DS-TCN) with cross-attention that ingests high-SNR contact PPG morphology, facial macro-foot timing, and the HRVP physiological anchor. The training objective must optimize a compound loss function:
$$\mathcal{L} = \mathcal{L}_{\text{Huber}} + \lambda_1 (1 - \text{Pearson}_r) + \lambda_2 \mathcal{L}_{\text{Variance}}$$
Quantize the model to 8-bit integers (INT8) via TFLite/ONNX for local mobile execution.

## Verification Resources & Methods
- **Reference Codebase**: Query and inspect modules in `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation` for baseline comparisons, POS filtering logic, and prior diagnostic findings.
- **Programmatic Verification**: Unit tests for sub-frame intersecting tangent foot detection, synthetic and real waveform synchronization jitter analysis, and batch evaluation against MIMIC / MCD-Iriun ground truth waveforms.
- **Diagnostic Template Collapse Audit**: Automated script computing Pearson correlation $r$, prediction distribution variance $\sigma^2(\hat{y}) / \sigma^2(y)$, and Bland-Altman agreement to verify absence of regression-to-the-mean.

## Acceptance Criteria

### Algorithmic & Diagnostic Standards
- [ ] Systolic Blood Pressure (SBP) MAE $\le 5.0\text{ mmHg}$ and Diastolic Blood Pressure (DBP) MAE $\le 3.5\text{ mmHg}$ on synchronized validation partitions.
- [ ] Pearson correlation coefficient $r \ge 0.82$ across diverse validation subjects (ruling out template collapse).
- [ ] Error Standard Deviation (SD) $\le 7.0\text{ mmHg}$, meeting AAMI/ISO 81060-2 and BHS Grade A criteria.

### Edge Device Hardware Feasibility (Poco M6 Pro 5G)
- [ ] Sustained 30 FPS concurrent dual-camera recording with zero dropped frames over a 10-second acquisition window.
- [ ] Fully local INT8 model inference time under $60\text{ ms}$ on the Snapdragon 4 Gen 2 mobile runtime with peak memory usage $\le 20\text{ MB}$.
