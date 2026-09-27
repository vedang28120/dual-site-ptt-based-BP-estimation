# Dispatch Assignment — Worker M3 (Milestone 3: Anti-Collapse Edge DL Architecture DS-TCN & INT8 Quantization)

## Context
You are `worker_m3` on the Dual-Site Blood Pressure Estimation Framework project.
Working directory: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m3`
Project root: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation`
Authoritative Request: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md`
Project Index: `C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md`

## Mandatory Reading
Read `ORIGINAL_REQUEST.md`, `PROJECT.md`, and Explorer handoffs:
- `.agents/explorer_survey_3/handoff.md` (Contains exact layer specifications, parameter count ~189.9k, FLOPs 53.7 MFLOPs, causal cross-attention kernel, tripartite compound loss $\mathcal{L} = \mathcal{L}_{\text{Huber}} + 0.40(1 - r) + 0.25\mathcal{L}_{\text{Variance}}$, and INT8 quantization strategy)
- `.agents/worker_m2/handoff.md` (Contains HRVP anchor contract and stiffness solver output)

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Exclusive Write Ownership
You exclusively own:
- `src/models/`
  - `src/models/__init__.py`
  - `src/models/ds_tcn.py`
  - `src/models/compound_loss.py`
  - `src/models/dataset.py`
  - `src/models/trainer.py`
  - `src/models/quantizer.py`
- `tests/unit/test_models.py`

DO NOT modify files outside your ownership.

## Mission & Specifications
1. Implement `ds_tcn.py`:
   - Anti-collapse Dual-Stream Temporal Convolutional Network with Causal Cross-Attention:
     - Stream A (Contact finger PPG morphology): 4x dilated causal Conv1D blocks ($d \in [1, 2, 4, 8]$, kernel size $K=5$, channels 32 -> 64). Receptive field $\approx 2.03$ seconds.
     - Stream B (Facial rPPG macro-foot timing): 4x dilated causal Conv1D blocks ($d \in [1, 2, 4, 8]$, kernel size $K=5$, channels 16 -> 32).
     - Multi-Head Causal Cross-Attention: $Q = H_{\text{morph}} W_Q$ ($[B, 300, 64]$), $K, V = H_{\text{timing}} W_{K, V}$ ($[B, 300, 32] \to 64$). Causal upper-triangular mask enforcing $j \le i$. Softmax attention distribution encodes pulse transit delay $\Delta PTT$.
     - HRVP Physiological Anchor MLP: Linear(6, 32) -> GELU -> Linear(32, 32). Ingests standardized 6D vector $z_{\text{HRVP}} = [\text{ePWV}, \Delta PTT, \Delta P_{\text{hydro}}, \frac{d(\text{ePWV})}{dP}, E_0, \alpha]$.
     - Bottleneck Fusion: Concatenates pooled temporal features (64) + HRVP anchor (32) = 96 channels.
     - Decoupled SBP and DBP heads: Linear(96, 48) -> GELU -> Linear(48, 1). Total parameters $\approx 189,980$, FLOPs $\approx 53.7$ MFLOPs.
2. Implement `compound_loss.py`:
   - Tripartite compound loss:
     $$\mathcal{L} = \mathcal{L}_{\text{Huber}} + \lambda_1 (1 - \text{Pearson}_r) + \lambda_2 \mathcal{L}_{\text{Variance}}$$
   - $\delta = 1.0\text{ mmHg}, \lambda_1 = 0.40, \lambda_2 = 0.25$.
   - $\mathcal{L}_{\text{Variance}} = |1 - \frac{\sigma(\hat{y})}{\sigma(y)}| + \max(0, 1 - \frac{\sigma^2(\hat{y})}{\sigma_{\text{threshold}}^2})$.
   - Mathematically penalizes variance collapse / regression-to-the-mean; gradient actively repels collapsed constant predictions.
3. Implement `dataset.py`:
   - PyTorch `DualSiteBPDataset` and batch collate functions handling synthetic and cached multi-subject pairs (morphology, timing, 6D HRVP anchor, SBP, DBP).
4. Implement `trainer.py`:
   - Training, validation, and evaluation loops computing MAE, RMSE, Pearson $r$, and prediction variance ratio $\sigma^2(\hat{y})/\sigma^2(y)$.
5. Implement `quantizer.py`:
   - Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) export.
   - Converts PyTorch model to ONNX (Opset 17) and INT8 TFLite flatbuffer (`dual_stream_tcn_int8.tflite`).
   - Profiles model size (< 250 KB target), MACs, and inference latency targeting Snapdragon 4 Gen 2 (< 60 ms budget).
6. Write unit tests in `tests/unit/test_models.py` verifying tensor shapes, receptive field, causal attention masking, compound loss anti-collapse gradients, training step convergence, and INT8 quantization export.
7. Run `pytest tests/unit/test_models.py`, verify all pass, document results in `handoff.md`.

## 2026-09-23T18:27:21Z
You are worker_m3 (Anti-Collapse Edge DL Architecture Specialist).
Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m3
Read C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\ORIGINAL_REQUEST.md, C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\PROJECT.md, and C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\worker_m3\DISPATCH.md.
Review Explorer 3 handoff in .agents/explorer_survey_3/handoff.md and Worker M2 handoff in .agents/worker_m2/handoff.md.
MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work.
Implement src/models/ (ds_tcn.py, compound_loss.py, dataset.py, trainer.py, quantizer.py, and __init__.py).
Implement unit tests in tests/unit/test_models.py.
Run pytest tests/unit/test_models.py, verify passing, document results in handoff.md. Notify orchestrator via send_message.

