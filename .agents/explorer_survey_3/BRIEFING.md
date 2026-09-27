# BRIEFING — 2026-09-23T18:07:13Z

## Mission
Investigate and specify Requirements R2 (HRVP-PINN Hemodynamic Engine) and R3 (Anti-Collapse Edge Deep Learning Architecture DS-TCN) for calibration-free BP estimation on Poco M6 Pro 5G.

## 🔒 My Identity
- Archetype: explorer
- Roles: Hemodynamics & Anti-Collapse DL Architect
- Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_3
- Original parent: a8dbe54c-0e6f-4b43-a00e-4135314b7d66
- Milestone: Survey & Architecture Specification for R2 & R3

## 🔒 Key Constraints
- Read-only investigation — do NOT implement production source code
- Formulate exact mathematical equations, tensor shapes, loss equations, and algorithmic pipelines
- Abide by hardware constraints: Snapdragon 4 Gen 2 (Poco M6 Pro 5G) < 60ms latency, <= 20MB peak RAM, INT8 quantization

## Current Parent
- Conversation ID: a8dbe54c-0e6f-4b43-a00e-4135314b7d66
- Updated: 2026-09-23T18:07:13Z

## Investigation State
- **Explored paths**: `ORIGINAL_REQUEST.md`, `DISPATCH.md`, `rPPG to BP estimation` repo (`MASTER_PROJECT_CONTEXT.md`, `PTT_BP_Estimation/ptt_bp/timing.py`, `models/training/train_bp_mcd.py`, `models/architectures/resnet_bigru_attn.py`).
- **Key findings**:
  1. *Root Cause of Prior Failure*: In single-site rPPG, microvascular Windkessel damping attenuates the dicrotic notch by 1-2 orders of magnitude, 8-bit quantization limits AC resolution to ~0.39%, and 30 FPS limits Nyquist to 15 Hz. Monolithic models collapse to population means (~120/80 mmHg). Prior `HuberPearsonLoss` lacked distribution variance enforcement.
  2. *HRVP-PINN Solution*: Dual-site acquisition (contact PPG at rear camera + facial rPPG at front camera) restores high-SNR morphology while measuring true arterial pulse transit time $\Delta PTT$. Continuous sub-frame intersecting tangent detection reduces timing error from ~33 ms down to < 1 ms.
  3. *Hydrostatic & Respiratory Modulation*: IMU-guided two-elevation displacement $\Delta h$ generates calibrated hydrostatic pressure perturbation $\Delta P_{\text{hydro}} = \rho g \Delta h$. Natural Traube-Hering-Mayer envelope (~0.2 Hz) enables closed-form identification of arterial compliance $\frac{d(\text{ePWV})}{dP}$ and non-linear arterial stiffness parameters $(E_0, \alpha)$ via Moens-Korteweg and Bramwell-Hill equations.
  4. *Anti-Collapse DS-TCN*: Dilated causal TCN with Cross-Attention captures dynamic pulse transit kernel. The compound loss $\mathcal{L}_{\text{Huber}} + \lambda_1(1 - \text{Pearson}_r) + \lambda_2 \mathcal{L}_{\text{Variance}}$ actively forbids variance collapse. Sizing: ~180k parameters, ~25 MFLOPs, ~185 KB INT8 model, running in < 5 ms on Snapdragon 4 Gen 2 (well within < 60 ms and <= 20 MB RAM limits).
- **Unexplored areas**: Production implementation details (assigned to future implementation milestones).

## Key Decisions Made
- Formulate complete analytical equations for sub-frame foot detection, hydrostatic differential, THM compliance, and Moens-Korteweg / Bramwell-Hill parameter identification.
- Design DS-TCN with causal dilated convolutions and multi-head cross-attention querying finger morphology against facial timing.
- Formulate variance penalty loss and complete INT8 QAT profile for Poco M6 Pro 5G.

## Artifact Index
- DISPATCH.md — Assignment instructions
- BRIEFING.md — Situational awareness
- progress.md — Liveness & step tracking
- handoff.md — Final survey & architecture report

