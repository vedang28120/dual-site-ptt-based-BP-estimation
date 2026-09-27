# Project Plan: Dual-Site Blood Pressure Estimation Framework

## Phase 0: Survey & Deep Analysis
- Spawn 3 parallel Explorers:
  1. `explorer_ref`: Deep probe of `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation` (extract baseline POS filtering, prior findings, diagnostic logs, failure modes, data formats).
  2. `explorer_optical`: Android Camera2 dual-stream architecture for Poco M6 Pro 5G (480p @ 30 FPS, flash LED contact PPG 64x64 center crop, MediaPipe face mesh ROI extraction, nanosecond `SENSOR_TIMESTAMP` hardware binding).
  3. `explorer_pinn_dl`: Mathematical formulations for HRVP-PINN ($D_{\text{path}}$, IMU hydrostatic perturbation $\Delta P = \rho g \Delta h$, Traube-Hering-Mayer respiratory modulation $d(\text{ePWV})/dP$, $E_0, \alpha$ identification) and anti-collapse DS-TCN architecture (cross-attention, compound loss with Huber + Pearson_r + Variance, INT8 quantization).
- Synthesize findings into `PROJECT.md` (Feature Inventory, Milestones, Interface Contracts, Code Layout).

## Phase 1: Dual-Track Architecture
### Track 1: E2E Testing Track (Independent Opaque-Box Suite)
- E2E Testing Orchestrator sets up `TEST_INFRA.md`.
- Build Tier 1 (Feature coverage), Tier 2 (Boundary & Corner cases), Tier 3 (Cross-feature combinations), Tier 4 (Real-world clinical & mobile workloads, MIMIC / MCD-Iriun benchmark).
- Build automated Template Collapse Audit script ($r \ge 0.82$, variance ratio $\sigma^2(\hat{y})/\sigma^2(y) \approx 1$, Bland-Altman, AAMI/BHS Grade A criteria).
- Publish `TEST_READY.md`.

### Track 2: Implementation Track
- **Milestone 1**: Low-Overhead Synchronized Optical Engine (Android Camera2 Dual-Stream, SENSOR_TIMESTAMP monotonic clock synchronization, 64x64 contact PPG extraction, MediaPipe face ROI tracker).
- **Milestone 2**: Physics-Informed Hemodynamic Engine (HRVP-PINN: sub-frame intersecting tangent foot detection, ePWV calculation, IMU hydrostatic tilt compensation, Traube-Hering-Mayer envelope extraction, calibration-free arterial stiffness $E_0, \alpha$ solver).
- **Milestone 3**: Anti-Collapse Edge Deep Learning Architecture (DS-TCN with cross-attention, compound loss optimizer, model training pipeline, TFLite/ONNX INT8 quantization < 60ms latency, <= 20MB RAM).
- **Milestone 4**: Mobile Integration & End-to-End Execution Pipeline (bridging optical engine, PINN solver, and INT8 TFLite runtime).

## Phase 2: Final Milestone & Hardening
- **Phase 2A**: Pass 100% of E2E tests (Tiers 1-4).
- **Phase 2B**: Adversarial Coverage Hardening (Tier 5) via Challengers + Forensic Integrity Audit (zero tolerance for template collapse, hardcoding, or dummy facades).

## Phase 3: Final Synthesis & Delivery
- Comprehensive verification report, benchmarks, documentation, and completion presentation.
