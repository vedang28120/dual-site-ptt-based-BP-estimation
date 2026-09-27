"""
Cloud Model Training Pipeline for Dual-Site BP Estimation.
Runs on Compute Engine GPU VM (1x NVIDIA T4 / L4) or Vertex AI Workbench.

Integrates:
- PulseDB Dataset Loader with polyphase 125 Hz -> 30 Hz resampling
- Strict Subject-Independent train/val/test splitting
- DualStreamTCN Anti-Collapse Causal Architecture
- Tripartite Compound Loss (Huber + Pearson Correlation + Variance Repulsion)
- Post-Training INT8 Quantization (Per-channel weight, asymmetric activation)
- Checkpoint persistence and GCS upload
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dataset import (
    PulseDBDataset,
    DualSiteBPDataset,
    collate_bp_batch,
    generate_synthetic_dataset,
    subject_independent_split,
)
from src.models.ds_tcn import DualStreamTCN
from src.models.compound_loss import TripartiteCompoundLoss
from src.models.trainer import BPTrainer, TrainingConfig
from src.models.quantizer import INT8Quantizer


def parse_args():
    parser = argparse.ArgumentParser(description="Cloud Training for Dual-Site BP Estimation")
    parser.add_argument("--data-path", type=str, default="./data/Train_Subset.mat", help="Path to PulseDB MAT subset")
    parser.add_argument("--test-data-path", type=str, default="./data/CalFree_Test_Subset.mat", help="Path to CalFree test subset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--use-synthetic", action="store_true", help="Fallback to synthetic dataset if MAT not found")
    parser.add_argument("--gcs-bucket", type=str, default="dual-site-bp-datasets-9c88cfa3", help="GCS bucket name for checkpoint sync")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("Dual-Site BP Estimation: Cloud Model Training Pipeline")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print("=" * 65)

    # 1. Dataset Loading
    if os.path.exists(args.data_path):
        print(f"[DATA] Loading PulseDB dataset from: {args.data_path}")
        train_val_dataset = PulseDBDataset.from_mat_file(args.data_path, target_fs=30.0)
        train_set, val_set, _ = subject_independent_split(train_val_dataset, train_ratio=0.85, val_ratio=0.15, test_ratio=0.0)
        
        if os.path.exists(args.test_data_path):
            print(f"[DATA] Loading Calibration-Free Test Set from: {args.test_data_path}")
            test_set = PulseDBDataset.from_mat_file(args.test_data_path, target_fs=30.0)
        else:
            test_set = val_set
    else:
        print(f"[DATA] MAT file '{args.data_path}' not found on disk.")
        print("[DATA] Initializing synthetic physiological generator (500 samples, 300 timesteps)...")
        synthetic_ds = generate_synthetic_dataset(num_samples=500, seq_len=300, seed=42)
        train_set, val_set, test_set = subject_independent_split(synthetic_ds, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    print(f"[DATA] Train Samples: {len(train_set)} | Val Samples: {len(val_set)} | Test Samples: {len(test_set)}")

    # 2. Model Initialization
    model = DualStreamTCN().to(device)

    total_params = model.count_parameters()
    print(f"[MODEL] DualStreamTCN initialized with {total_params:,} parameters (Budget: <250,000)")
    assert total_params < 250_000, f"Model parameters ({total_params}) exceed edge deployment limit!"

    # 3. Training Loop Setup
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        delta=1.0,
        lambda_1=0.40,
        lambda_2=0.25,
        patience=15,
        device=str(device),
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_bp_batch)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_bp_batch)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_bp_batch)

    trainer = BPTrainer(
        model=model,
        config=config,
    )

    print("\n[TRAIN] Beginning training with Tripartite Compound Loss...")
    history = trainer.fit(train_loader, val_loader, verbose=True)

    # 4. Final Evaluation on Calibration-Free Test Set
    print("\n[EVAL] Running final evaluation on unseen subject test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("=" * 65)
    print("Final Model Benchmark Results:")
    print(f"  SBP MAE       : {test_metrics.sbp_mae:.2f} mmHg")
    print(f"  DBP MAE       : {test_metrics.dbp_mae:.2f} mmHg")
    print(f"  SBP RMSE      : {test_metrics.sbp_rmse:.2f} mmHg")
    print(f"  DBP RMSE      : {test_metrics.dbp_rmse:.2f} mmHg")
    print(f"  SBP Pearson r : {test_metrics.sbp_pearson_r:.4f}")
    print(f"  DBP Pearson r : {test_metrics.dbp_pearson_r:.4f}")
    print(f"  AAMI Pass     : {test_metrics.aami_pass}")
    print(f"  Collapse Alert: {test_metrics.template_collapse_detected}")
    print("=" * 65)

    # 5. Checkpoint & Edge Export
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"[SAVE] Best PyTorch weights saved to: {checkpoint_path}")

    # Quantization and Edge Profiling
    print("[QUANT] Profiling and exporting INT8 model for Snapdragon 4 Gen 2...")
    quantizer = INT8Quantizer(model)
    profile = quantizer.profile_model()
    print(f"[PROFILE] Latency: {profile['estimated_cpu_latency_ms']:.1f} ms | RAM: {profile['estimated_peak_memory_mb']:.1f} MB | INT8 Size: {profile['estimated_int8_model_size_kb']:.1f} KB")

    onnx_path = os.path.join(args.output_dir, "dual_stream_tcn.onnx")
    onnx_res = quantizer.export_onnx(onnx_path)
    print(f"[EXPORT] ONNX model saved to: {onnx_res['save_path']} ({onnx_res['file_size_kb']:.1f} KB)")

    tflite_path = os.path.join(args.output_dir, "dual_stream_tcn_int8.tflite")
    tflite_res = quantizer.export_tflite_int8(tflite_path)
    print(f"[EXPORT] INT8 TFLite model saved to: {tflite_res['save_path']} ({tflite_res['file_size_kb']:.1f} KB, SNR: {tflite_res['mean_snr_db']:.1f} dB)")

    # 6. GCS Upload if available
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(args.gcs_bucket)
        if bucket.exists():
            print(f"[GCS] Syncing artifacts to gs://{args.gcs_bucket}/checkpoints/...")
            for fname in ["best_model.pt", "dual_stream_tcn.onnx", "dual_stream_tcn_int8.tflite"]:
                fpath = os.path.join(args.output_dir, fname)
                if os.path.exists(fpath):
                    blob = bucket.blob(f"checkpoints/{fname}")
                    blob.upload_from_filename(fpath)
                    print(f"      + Uploaded gs://{args.gcs_bucket}/checkpoints/{fname}")
    except Exception as e:
        print(f"[GCS] GCS sync skipped ({e}). Models stored locally at: {args.output_dir}")

    print("\n[COMPLETE] Cloud training and quantization pipeline executed successfully!")


if __name__ == "__main__":
    main()
