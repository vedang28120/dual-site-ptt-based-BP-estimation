# PulseFusion: Dual-Site Cuffless Blood Pressure Estimation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Android](https://img.shields.io/badge/Platform-Android%20Native-green.svg)](mobile/pulsefusion_app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-282%20Passed-brightgreen.svg)](tests/)

A calibration-free, cuffless blood pressure screening system that couples synchronized dual-site optical signals (facial remote photoplethysmography [rPPG] + fingertip contact photoplethysmography [PPG]) with inertial sensing and physics-informed neural network dynamics. Designed for real-time on-device inference on commodity smartphones.

---

## 🌟 Key Innovations

1. **Dual-Camera Optical Differential Pulse Transit Time (dPTT)**:
   - **Front Camera**: Extracts non-contact facial rPPG waveforms via real-time ROI tracking.
   - **Rear Camera + Torch**: Captures high-SNR fingertip contact PPG waveforms.
   - **Differential Arrival**: The arrival time delta eliminates cardiac pre-ejection period (PEP) variability without requiring ECG leads.

2. **Hydrostatic Perturbation Engine (HRVP-PINN)**:
   - Uses the smartphone's 6-axis IMU (accelerometer + gyroscope) to track arm elevation changes ($\Delta h$).
   - Induces known hydrostatic pressure offsets ($\Delta P_{\text{hydro}} = -\rho g \Delta h$) to actively calibrate patient-specific arterial compliance ($E_0, \alpha$) in a closed-form physics solver.

3. **Anti-Collapse Dual-Stream Temporal Convolutional Network (DS-TCN)**:
   - Asymmetric dual-stream encoder with causal cross-attention between facial and finger pulse dynamics.
   - Tripartite compound loss with variance preservation and anti-collapse gradient repulsion.

4. **PulseFormer Foundation & Edge Distillation Architecture**:
   - **PulseFormer-400M**: Multi-scale foundation teacher network with bi-directional cross-attention and physiological hint projections.
   - **PulseFormer-Lite-50M**: Distilled edge student model quantized to INT8 with native ONNX Runtime integration for sub-30ms mobile execution on Snapdragon 4 Gen 2 (SM4450).

---

## 📐 System Architecture

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

---

## 📂 Project Structure

```
dual_site_bp_estimation/
├── checkpoints/                    # Lightweight model configurations and audit reports
├── docs/                           # Architecture, Colab guides, roadmap & presentation
│   ├── presentation/               # Slide decks and project summaries
│   ├── ARCHITECTURE.md             # Complete technical architecture report
│   ├── COLAB_HANDOFF.md            # Cloud GPU training & dataset handoff guide
│   └── ROADMAP.md                  # Project roadmap & future workflow milestones
├── mobile/                         # Mobile edge client implementation
│   └── pulsefusion_app/            # Native Android Studio project (Kotlin + Chaquopy + ONNX)
│       ├── app/src/main/java/      # Camera2 dual-stream capture, trackers & inference engine
│       ├── app/src/main/python/    # Embedded signal processing & PINN solvers
│       └── app/src/main/assets/    # Edge models (INT8 TFLite, MediaPipe tasks, weights)
├── notebooks/
│   └── dual_site_ptt_to_bp_model.ipynb # Interactive training, validation, and analysis notebook
├── scripts/                        # Model generation, Colab exports & verification
│   ├── generate_colab_notebook.py
│   └── export_trained_notebook_models.py
├── src/                            # Core Python package
│   ├── models/                     # PyTorch models: DS-TCN, PulseFormer, compound loss & quantizer
│   ├── optical/                    # Dual-camera sync, PCHIP resampler, ROI tracker & filters
│   ├── pinn/                       # HRVP-PINN solver, foot detector, anthropometrics & registration
│   └── pipeline/                   # Unified clinical evaluation and training pipeline
├── tests/                          # Comprehensive unit and integration test suite (282 tests)
├── requirements.txt                # Python dependencies
└── pytest.ini                      # Pytest configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- Android Studio Hedgehog / Iguana (for mobile deployment)
- Android device with camera + flashlight support (tested on Qualcomm Snapdragon 4 Gen 2 / ARM64-v8a)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/vedang28120/dual-site-ptt-based-BP-estimation.git
cd dual-site-ptt-based-BP-estimation

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Running the Test Suite

Run the full pytest suite (282 tests covering PINN solvers, optical resamplers, neural models, and end-to-end pipelines):

```bash
pytest tests/ -v
```

---

## 📱 Mobile App (PulseFusion BP)

The native Android application is located in `mobile/pulsefusion_app/`.

### Architecture:
- **Dual Camera2 Pipeline**: Synchronized capture of rear camera (contact PPG with torch locked at $3200\text{K}$) and front camera (facial rPPG at $1920\times1080$).
- **Native ONNX Runtime**: Uses `ai.onnxruntime:onnxruntime-android` to execute `pulseformer_lite_50m_int8.onnx` via ARM64 NEON SIMD kernels.
- **Embedded Python (Chaquopy)**: Executes the exact Python-based sub-frame intersecting tangent foot detector and HRVP-PINN solver on device.

### Build and Deploy:
Open `mobile/pulsefusion_app` in Android Studio, connect your Android device via USB debugging, and click **Run** (`Shift + F10`).

---

## 📊 Model Zoo & Quantization

| Model Architecture | Parameters | Input Representation | Target Metrics | Format |
|---|---|---|---|---|
| **DualStreamTCN Compact** | ~480K | BVP Face + BVP Finger (125 Hz) | SBP, DBP | INT8 TFLite |
| **DualStreamTCN XL** | ~2.47M | Multi-channel BVP + VPG + APG | SBP, DBP | INT8 TFLite / NumPy |
| **PulseFormer-400M** | ~400.3M | Multimodal Face + Palm + 14D Physio | Continuous BP Waveform | PyTorch FP32 / FP16 |
| **PulseFormer-Lite-50M** | ~50.0M | Multimodal Face + Palm + 14D Physio | SBP, DBP, MAP | INT8 ONNX (Mobile) |

To export and quantize models for mobile deployment:
```bash
python scripts/export_trained_notebook_models.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
