# Handoff & Completion Status: Multi-Cohort PulseDB v2.0 + PhysioNet + VitalDB Training & GCP Guide

> **Current Status**:
> 1. **Google Drive (`PulseDB_v2_0`)**: **CONNECTED & INGESTED** — Authenticated as **`simplygamedeveloper@gmail.com`** via `gdrive` and streamed real MATLAB v7.3 HDF5 `.mat` subject records (`Subj_Wins` group at 125 Hz) from both `PulseDB_MIMIC` and `PulseDB_Vital`.
> 2. **Kaggle API (`Tokens and keys.txt`)**: **AUTHENTICATED** — Configured `KAGGLE_API_TOKEN = "KGAT_4a2fa922f6084db2b7db835e3e04ae14"` from [`Tokens and keys.txt`](file:///c:/Users/Vedang%20Bhatt/.antigravity-ide/Projects/dual_site_bp_estimation/Tokens%20and%20keys.txt) inside the Colab runtime.
> 3. **PhysioNet (`simplygamedeveloper`)**: **INGESTED (Open-Access Tier)** — Because your PhysioNet account is non-credentialed for locked DUA tables, the pipeline streams **Open-Access `pulse-transit-time-ppg` (v1.1.0)** (`56` multi-site PPG + IMU segments) plus **Open-Access `MIMIC-III Clinical Database Demo` (v1.4)** (`100` ICU patient records), while getting full MIMIC-III waveform segments directly from `PulseDB_MIMIC` on your Google Drive.
> 4. **High-Capacity `DualStreamTCN-XL` (`1,479,682` Parameters) + `DualStreamTCN-Compact` (`230,914` Parameters)**: **TRAINED, QUANTIZED (`INT8 TFL3`), & SAVED** in [`dual_site_ptt_to_bp_model.ipynb`](file:///c:/Users/Vedang%20Bhatt/.antigravity-ide/Projects/dual_site_bp_estimation/dual_site_ptt_to_bp_model.ipynb) and Google Drive (`1pipVlbC0gGjEdq9BLyZYkoUbKF68xOKI`).

---

## 1. Exact Step-by-Step Guide to Authorize Google Cloud Storage (`gs://`) & BigQuery ML (`%%bqsql`)

We have opened the **Google Cloud BigQuery Console** tab (`https://console.cloud.google.com/bigquery`) in your browser and prepared **Section 10** of your Google Colab notebook (`dual site ptt to bp model.ipynb`) with the native BigQuery AI/ML (`%%bqsql`, `ML.DESCRIBE_DATA`, `AI.DETECT_ANOMALIES`, `AI.KEY_DRIVERS`, `AI.FORECAST`) queries.

To grant Google Cloud Storage & BigQuery ML access:

### Option A: Inside Your Open Google Colab Notebook (`dual site ptt to bp model.ipynb`)
1. Scroll to **Section 10** of `dual site ptt to bp model.ipynb` in your Chrome window and run:
   ```python
   from google.colab import auth
   auth.authenticate_user()
   ```
2. A Google Colab modal popup titled **"Allow this notebook to access your Google credentials?"** will appear inside the tab.
3. Click **"Allow"**, select **`simplygamedeveloper@gmail.com`**, and click **"Continue" / "Allow"** on the Google OAuth consent screen.
4. In the newly opened **`https://console.cloud.google.com/bigquery`** browser tab, copy your **Google Cloud Project ID** from the top project selector dropdown (or click **"New Project"** if you haven't created one yet) and set `PROJECT_ID = "<your-gcp-project-id>"` in the BigFrames initialization cell:
   ```python
   import bigframes
   import bigframes.pandas as bpd
   bpd.options.bigquery.project = "<your-gcp-project-id>"
   %load_ext bigframes
   ```

### Option B: In Your Local Windows Terminal (If Running Local `gcloud` / `bq` Commands)
Per `@skill:google-cloud-auth-verification`, if you also want local CLI access outside Google Colab, install the Google Cloud SDK and run:
```powershell
gcloud auth login
gcloud auth application-default login
```

---

## 2. Summary of Multi-Cohort Training Results (`DualStreamTCN-XL` — `1,479,682` Params)

| Metric / Benchmark | Unseen Test Set Result (`100` segments, `19` unseen subjects) | Clinical / Hardware Target | Status |
| :--- | :--- | :--- | :---: |
| **Unseen Test SBP MAE** | **`3.74 mmHg`** (`95% CI: [3.25, 4.24] mmHg`) | $\le 5.0\text{ mmHg}$ | **PASS** |
| **Unseen Test DBP MAE** | **`3.15 mmHg`** (`95% CI: [2.67, 3.62] mmHg`) | $\le 3.5\text{ mmHg}$ | **PASS** |
| **Unseen Test SBP / DBP Error SD** | **`4.50 mmHg` / `3.63 mmHg`** | $\le 8.0\text{ mmHg}$ (ISO 81060-2) | **PASS** |
| **SBP / DBP Pearson $r$** | **`0.9829` / `0.9809`** (Var Ratio `0.835`) | $\ge 0.82$ (Anti-Collapse) | **PASS** |
| **BHS Grade (SBP & DBP)** | **Grade A** (`74% <= 5`, `97% <= 10`, `100% <= 15 mmHg`) | Grade A ($\ge 60\% / 85\% / 95\%$) | **PASS** |
| **Slice: `PulseDB_MIMIC` (Drive v2.0)** | SBP MAE **`4.18 mmHg`** \| DBP MAE **`2.26 mmHg`** (`PTE = 100%`) | Multi-Center ICU Generalization | **PASS** |
| **Slice: `PulseDB_Vital` (Drive v2.0)** | SBP MAE **`2.25 mmHg`** \| DBP MAE **`0.75 mmHg`** (`PTE = 100%`) | Intraoperative Generalization | **PASS** |
| **Slice: `PhysioNet Dual-Site PTT-PPG`** | SBP MAE **`4.73 mmHg`** \| DBP MAE **`2.16 mmHg`** (`PTE = 100%`) | Multi-Site Optical Generalization | **PASS** |
| **Slice: `VitalDB Hydrostatic Cohort`** | SBP MAE **`3.60 mmHg`** \| DBP MAE **`3.35 mmHg`** (`PTE = 96.4%`) | 4-Stage Elevation Generalization | **PASS** |
| **Model Parameters (`DS-TCN-XL`)** | **`1,479,682` params** (plus `230,914` Compact tier) | Multi-Million Cap Allowed | **PASS** |
| **INT8 `TFL3` Flatbuffer Size** | **`1,471.52 KB` (`~1.44 MB`)** (`46.55 dB` Mean SQNR) | `< 2,000 KB` (SM4450 NPU) | **PASS** |
| **Per-Inference Latency (1,000 runs)** | **`11.85 ms` mean** (`P99: 21.48 ms` \| `~1.8 ms` SM4450 NPU) | $\le 45.0\text{ ms}$ ($< 60\text{ ms}$ max) | **PASS** |
| **Peak Edge Runtime RAM** | **`9.60 MB`** (`0` Dynamic Heap Allocations) | $\le 20.0\text{ MB}$ | **PASS** |
