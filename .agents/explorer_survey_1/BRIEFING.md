# BRIEFING — 2026-09-23T18:15:00Z

## Mission
Investigate and survey reference codebase C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation (algorithms, pipelines, models, failure modes, data structures, template collapse) to inform the dual-site BP estimation framework.

## 🔒 My Identity
- Archetype: explorer
- Roles: Reference Codebase & Baseline Surveyor
- Working directory: C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_1
- Original parent: a8dbe54c-0e6f-4b43-a00e-4135314b7d66
- Milestone: Reference Codebase & Baseline Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Strictly read-only access to C:\Users\Vedang Bhatt\.antigravity-ide\Projects\rPPG to BP estimation
- Write only to C:\Users\Vedang Bhatt\.antigravity-ide\Projects\dual_site_bp_estimation\.agents\explorer_survey_1
- Do not modify source code or reference projects

## Current Parent
- Conversation ID: a8dbe54c-0e6f-4b43-a00e-4135314b7d66
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `MASTER_PROJECT_CONTEXT.md`, `rppg_to_bp_master_roadmap-v2.md`, `LuminaBP_Vocational_Training_Report.md`
  - `core_extraction/` (`pos_extractor.py`, `chrom_extractor.py`, `green_extractor.py`, `face_mesh_tracker.py`, `multi_patch_cca.py`)
  - `filtering/` (`temporal_resampler.py`, `dual_stream_filter.py`, `wavelet_denoiser.py`, `detrending.py`, `sqi_validator.py`, `topological_mai.py`)
  - `models/architectures/` (`resnet_bigru_attn.py`, `drp_bbp_net.py`, `bayesian_ufacebp.py`, `demo_mlp.py`, `legacy_lstm.py`)
  - `models/training/` (`train_bp_mcd.py`, `make_notebook.py`)
  - `PTT_BP_Estimation/` (`timing.py`, `synchronization.py`, `estimation.py`, `features.py`, `data_loading.py`)
  - `prototype_tscan_bp/` (`bp_estimator.py`, `evaluate_mcd_1020.py`)
  - `camera pipeline/` & `mobile/android_app/` (`CameraService.kt`, `FaceLandmarkTracker.kt`, `BPInferenceEngine.kt`, `pos_engine.py`)
  - `results/` & `camera pipeline/tests/` (`mcd_1020_accuracy_comparison.json`, `balancing_comparison.csv`, `gold_pipeline_ablation_report.md`, `accuracy_report_50subjects.md`, `accuracy_report_lstm_direct_ppg.md`)
- **Key findings**:
  - Identified smoking gun evidence of Template Collapse: all methods (TS-CAN, POS, CHROM, Green) predict SBP ~127.1 mmHg and DBP ~76.3 mmHg regardless of whether subject is resting (GT 105/78) or post-exercise (GT 113/78).
  - 50-subject benchmark showed negative correlation ($r = -0.311$ for SBP, $r = -0.027$ for DBP) and predicted range only 37% of true range (31.4 mmHg vs 85.0 mmHg).
  - Even on direct contact PPG (`.PW`), standalone LSTM yielded $r = -0.205$, proving single-site optical morphology lacks physiological identifiability without transit dynamics.
  - Smoothness Priors ($\lambda=100$) removes 0.1-0.4 Hz respiratory modulation, inadvertently deleting the Traube-Hering-Mayer envelope needed for compliance calculation.
  - MediaPipe landmarking in Android app converts YUV->NV21->JPEG->Bitmap on CPU, causing severe frame drops when frame analysis exceeds 33ms budget.
  - PTT baseline in `PTT_BP_Estimation` uses discrete integer sample `argmin`, failing sub-frame precision at 30 FPS.
- **Unexplored areas**: None in reference codebase; ready for synthesis and handoff.

## Key Decisions Made
- Structured complete handoff report following 5-component protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.

## Artifact Index
- DISPATCH.md — Assignment instructions
- BRIEFING.md — Working memory and survey findings
- progress.md — Heartbeat and milestone tracker
- handoff.md — Comprehensive reference codebase survey and baseline report
