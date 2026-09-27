"""
Comprehensive Unit Test Suite for Anti-Collapse Edge DL Architecture (Milestone 3).

Verifies:
  1. DS-TCN model architecture, parameter count budget (< 250k), layer dimensions, and receptive field (~2.03s).
  2. Strict causal convolution and causal cross-attention upper-triangular masking.
  3. Dynamic temporal sequence length handling (e.g. T=150, T=300).
  4. Decoupled SBP and DBP regression heads and gradient backpropagation.
  5. Tripartite compound loss mathematical components (Huber, Pearson, Variance) and anti-collapse gradient repulsion.
  6. Dataset generation, indexing, batch collation, and train/val/test splitting.
  7. Training step execution, validation evaluation, metric calculation, and checkpoint persistence.
  8. Symmetric per-channel weight and asymmetric activation INT8 quantization, SNR preservation (>= 35 dB).
  9. Mobile edge runtime profiler for Qualcomm Snapdragon 4 Gen 2 (< 60 ms budget, <= 20 MB RAM, < 250 KB file).
  10. ONNX (Opset 17) and INT8 TFLite flatbuffer export.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ds_tcn import (
    DualStreamTCN,
    CausalConv1d,
    DilatedCausalBlock,
    CausalCrossAttention,
)
from src.models.compound_loss import (
    TripartiteCompoundLoss,
    PyTorchTripartiteLoss,
    NumpyTripartiteLoss,
    ReferenceTripartiteLoss,
)
from src.models.dataset import (
    DualSiteBPDataset,
    PulseDBDataset,
    collate_bp_batch,
    generate_synthetic_dataset,
    train_val_test_split,
    subject_independent_split,
)
from src.models.trainer import (
    BPTrainer,
    TrainingConfig,
    EvaluationMetrics,
)
from src.models.quantizer import (
    INT8Quantizer,
    ReferenceINT8Quantizer,
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    dequantize_asymmetric,
    calculate_quantization_error_snr,
    FakeQuantizeWeight,
)


# ============================================================================
# 1. DS-TCN Architecture Tests
# ============================================================================
class TestDsTcnArchitecture:
    """Verifies DS-TCN neural network architecture and causal mechanics."""

    def test_parameter_count_under_edge_budget(self):
        """Model must contain under 250k parameters for edge deployment."""
        model = DualStreamTCN()
        total_params = model.count_parameters()
        assert total_params < 250_000, f"Parameter count {total_params} exceeds 250k budget"
        assert total_params > 100_000, f"Parameter count {total_params} is suspiciously low"

    def test_receptive_field_covers_cardiac_cycles(self):
        """Receptive field must be exactly 61 samples (2.03 seconds at 30 Hz)."""
        rf = DualStreamTCN.receptive_field()
        assert rf == 61
        rf_seconds = rf / 30.0
        assert 2.0 <= rf_seconds <= 2.1

    def test_causal_conv1d_strict_causality(self):
        """Future inputs must not influence past outputs in CausalConv1d."""
        conv = CausalConv1d(in_channels=1, out_channels=4, kernel_size=5, dilation=2)
        conv.eval()

        # Create input of length 20
        x1 = torch.randn(1, 1, 20)
        out1 = conv(x1)

        # Alter time step 15
        x2 = x1.clone()
        x2[0, 0, 15] += 10.0
        out2 = conv(x2)

        # Steps 0 through 14 must be bit-identical
        assert torch.allclose(out1[:, :, :15], out2[:, :, :15], atol=1e-6)
        # Step 15 and beyond may differ
        assert not torch.allclose(out1[:, :, 15:], out2[:, :, 15:], atol=1e-4)

    def test_dilated_causal_block_residual_connection(self):
        """DilatedCausalBlock must preserve spatial and temporal dimensions with residual addition."""
        block = DilatedCausalBlock(channels=64, kernel_size=5, dilation=4)
        block.eval()
        x = torch.randn(2, 64, 50)
        out = block(x)
        assert out.shape == (2, 64, 50)
        assert not torch.isnan(out).any()

    def test_causal_cross_attention_strict_masking(self):
        """Attention weights must be strictly 0 for future tokens (j > i)."""
        attn = CausalCrossAttention(query_dim=64, key_dim=32, num_heads=4)
        attn.eval()
        q = torch.randn(1, 30, 64)
        kv = torch.randn(1, 30, 32)
        out, weights = attn(q, kv)

        assert out.shape == (1, 30, 64)
        assert weights.shape == (1, 4, 30, 30)

        # Check upper-triangle of attention weights is zero
        w_np = weights[0, 0].detach().cpu().numpy()
        for i in range(30):
            for j in range(i + 1, 30):
                assert w_np[i, j] == 0.0, f"Non-causal leakage at position ({i}, {j}): {w_np[i, j]}"

    def test_forward_pass_dual_output_shapes(self):
        """Model must output SBP and DBP of shape [B, 1]."""
        model = DualStreamTCN()
        model.eval()
        x_morph = torch.randn(4, 1, 300)
        x_timing = torch.randn(4, 1, 300)
        z_anchor = torch.randn(4, 6)

        sbp, dbp = model(x_morph, x_timing, z_anchor)
        assert sbp.shape == (4, 1)
        assert dbp.shape == (4, 1)
        assert not torch.isnan(sbp).any()
        assert not torch.isnan(dbp).any()

    def test_dynamic_sequence_length_robustness(self):
        """Model must gracefully process variable sequence lengths (T=100, T=150, T=300)."""
        model = DualStreamTCN()
        model.eval()
        for seq_len in [100, 150, 300]:
            x_m = torch.randn(2, 1, seq_len)
            x_t = torch.randn(2, 1, seq_len)
            z = torch.randn(2, 6)
            sbp, dbp = model(x_m, x_t, z)
            assert sbp.shape == (2, 1)
            assert dbp.shape == (2, 1)

    def test_decoupled_prediction_heads_independence(self):
        """SBP and DBP heads must have distinct parameter sets and generate distinct predictions."""
        model = DualStreamTCN()
        # Verify distinct weights
        assert model.sbp_head[0].weight.data_ptr() != model.dbp_head[0].weight.data_ptr()

        x_m = torch.randn(2, 1, 300)
        x_t = torch.randn(2, 1, 300)
        z = torch.randn(2, 6)
        sbp, dbp = model(x_m, x_t, z)
        # SBP and DBP should not be identical
        assert not torch.allclose(sbp, dbp)

    def test_gradient_flow_end_to_end(self):
        """Gradients must propagate back through both optical streams and 6D anchor."""
        model = DualStreamTCN()
        x_m = torch.randn(2, 1, 100, requires_grad=True)
        x_t = torch.randn(2, 1, 100, requires_grad=True)
        z = torch.randn(2, 6, requires_grad=True)

        sbp, dbp = model(x_m, x_t, z)
        loss = sbp.sum() + dbp.sum()
        loss.backward()

        assert x_m.grad is not None and not torch.allclose(x_m.grad, torch.zeros_like(x_m.grad))
        assert x_t.grad is not None and not torch.allclose(x_t.grad, torch.zeros_like(x_t.grad))
        assert z.grad is not None and not torch.allclose(z.grad, torch.zeros_like(z.grad))

    def test_attention_weights_inspection(self):
        """get_attention_weights must return attention map with shape [B, num_heads, T, T]."""
        model = DualStreamTCN()
        x_m = torch.randn(2, 1, 40)
        x_t = torch.randn(2, 1, 40)
        attn_map = model.get_attention_weights(x_m, x_t)
        assert attn_map.shape == (2, 4, 40, 40)


# ============================================================================
# 2. Tripartite Compound Loss Tests
# ============================================================================
class TestTripartiteCompoundLoss:
    """Verifies tripartite compound loss math, variance anti-collapse, and gradient repulsion."""

    def test_huber_quadratic_and_linear_regions(self):
        """Huber loss must be quadratic for error <= delta and linear for error > delta."""
        loss_fn = TripartiteCompoundLoss(delta=1.0)
        # Small error: 0.5 * 0.4^2 = 0.08
        y_true = torch.tensor([120.0])
        y_pred = torch.tensor([120.4])
        _, metrics = loss_fn(y_pred, y_true)
        assert abs(metrics["loss_huber"] - 0.08) < 1e-4

        # Large error: delta * (|e| - 0.5 * delta) = 1.0 * (3.0 - 0.5) = 2.5
        y_pred_large = torch.tensor([123.0])
        _, metrics_large = loss_fn(y_pred_large, y_true)
        assert abs(metrics_large["loss_huber"] - 2.5) < 1e-4

    def test_pearson_component_bounds(self):
        """Pearson loss must be 0 for perfect correlation and 2 for negative correlation."""
        loss_fn = TripartiteCompoundLoss()
        y_true = torch.tensor([100.0, 120.0, 140.0, 160.0])

        # Perfect positive correlation (r = 1.0 -> loss = 0.0)
        _, met_pos = loss_fn(y_true * 2.0 + 10.0, y_true)
        assert met_pos["pearson_r"] > 0.999
        assert met_pos["loss_pearson"] < 1e-4

        # Perfect negative correlation (r = -1.0 -> loss = 2.0)
        _, met_neg = loss_fn(-y_true, y_true)
        assert met_neg["pearson_r"] < -0.999
        assert abs(met_neg["loss_pearson"] - 2.0) < 1e-4

    def test_variance_collapse_penalty(self):
        """Collapsed constant predictions must trigger high variance penalty."""
        loss_fn = TripartiteCompoundLoss(sigma_thresh=8.0)
        y_true = torch.tensor([100.0, 120.0, 140.0, 160.0])
        y_collapsed = torch.tensor([130.0, 130.0, 130.0, 130.0])

        _, met = loss_fn(y_collapsed, y_true)
        assert met["loss_variance"] > 1.5
        assert met["std_pred"] < 0.1

    def test_anti_collapse_gradient_repulsion(self):
        """Gradient on collapsed predictions must actively push them away from the mean."""
        loss_fn = TripartiteCompoundLoss()
        y_true = torch.tensor([100.0, 120.0, 140.0, 160.0])
        y_pred = torch.tensor([130.0, 130.0, 130.0, 130.0], requires_grad=True)

        loss, _ = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None
        # Must not be zero
        assert not torch.allclose(y_pred.grad, torch.zeros_like(y_pred))
        # The gradient must push the first prediction lower (negative derivative)
        # and last prediction higher
        assert y_pred.grad[0] > 0.0  # Loss decreases as pred[0] decreases (towards 100)
        assert y_pred.grad[-1] < 0.0  # Loss decreases as pred[-1] increases (towards 160)

    def test_dual_target_sbp_dbp_loss(self):
        """Loss function must handle [B, 2] inputs representing both SBP and DBP."""
        loss_fn = TripartiteCompoundLoss()
        y_true = torch.tensor([[120.0, 80.0], [140.0, 90.0], [110.0, 70.0], [160.0, 100.0]])
        y_pred = torch.tensor([[122.0, 78.0], [138.0, 92.0], [112.0, 71.0], [158.0, 98.0]])

        loss, metrics = loss_fn(y_pred, y_true)
        assert loss.item() > 0.0
        assert "sbp_pearson_r" in metrics
        assert "dbp_pearson_r" in metrics
        assert "sbp_variance_ratio" in metrics
        assert "dbp_variance_ratio" in metrics

    def test_numpy_and_pytorch_parity(self):
        """NumPy reference and PyTorch module must agree within numerical precision."""
        py_loss = TripartiteCompoundLoss(delta=1.0, lambda_1=0.40, lambda_2=0.25)
        np_loss = NumpyTripartiteLoss(delta=1.0, lambda_1=0.40, lambda_2=0.25)

        y_true = np.array([105.0, 122.0, 138.0, 155.0])
        y_pred = np.array([108.0, 119.0, 141.0, 152.0])

        res_np = np_loss.compute_loss(y_true, y_pred)
        _, res_py = py_loss(torch.tensor(y_pred), torch.tensor(y_true))

        assert abs(res_np["loss_huber"] - res_py["loss_huber"]) < 1e-4
        assert abs(res_np["loss_pearson"] - res_py["loss_pearson"]) < 1e-4
        assert abs(res_np["loss_variance"] - res_py["loss_variance"]) < 1e-4
        assert abs(res_np["total_loss"] - res_py["loss_total"]) < 1e-4


# ============================================================================
# 3. Dataset and DataLoader Tests
# ============================================================================
class TestDatasetAndLoader:
    """Verifies dataset generation, formatting, collation, and splitting."""

    def test_synthetic_dataset_generation(self):
        """Synthetic generator must create valid multi-subject pairs with 6D anchors."""
        ds = generate_synthetic_dataset(num_samples=25, seq_len=300, seed=123)
        assert len(ds) == 25

        item = ds[0]
        assert item["morph"].shape == (1, 300)
        assert item["timing"].shape == (1, 300)
        assert item["anchor"].shape == (6,)
        assert item["sbp"].shape == (1,)
        assert item["dbp"].shape == (1,)
        assert item["targets"].shape == (2,)

    def test_collate_function_batching(self):
        """collate_bp_batch must produce properly sized batch tensors."""
        ds = generate_synthetic_dataset(num_samples=10, seq_len=150, seed=42)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_bp_batch)

        batch = next(iter(loader))
        assert batch["morph"].shape == (4, 1, 150)
        assert batch["timing"].shape == (4, 1, 150)
        assert batch["anchor"].shape == (4, 6)
        assert batch["targets"].shape == (4, 2)
        assert len(batch["subject_ids"]) == 4

    def test_train_val_test_split_integrity(self):
        """Dataset splitting must produce disjoint partitions with expected ratios."""
        ds = generate_synthetic_dataset(num_samples=40, seq_len=100, seed=99)
        train_set, val_set, test_set = train_val_test_split(ds, 0.70, 0.15, 0.15, seed=99)

        assert len(train_set) == 28
        assert len(val_set) == 6
        assert len(test_set) == 6

        # Check disjoint indices
        all_indices = set(train_set.indices).union(val_set.indices).union(test_set.indices)
        assert len(all_indices) == 40

    def test_invalid_dimension_validation(self):
        """Mismatched sample counts must raise ValueError."""
        with pytest.raises(ValueError):
            DualSiteBPDataset(
                morph_signals=np.zeros((10, 1, 100)),
                timing_signals=np.zeros((10, 1, 100)),
                anchor_vectors=np.zeros((10, 6)),
                sbp_targets=np.zeros((9, 1)),  # Mismatch!
                dbp_targets=np.zeros((10, 1))
            )

    def test_pulsedb_dataset_from_subset_dict(self):
        """PulseDBDataset must ingest 125 Hz multi-channel arrays, resample to 30 Hz, and construct 6D anchors."""
        rng = np.random.RandomState(42)
        n_samples = 6
        # Signals: 3 channels (ECG_F, PPG_F, ABP_Raw), 1250 samples @ 125 Hz
        signals = rng.randn(n_samples, 3, 1250).astype(np.float32)
        sbp = np.array([[120.0], [125.0], [130.0], [135.0], [140.0], [145.0]], dtype=np.float32)
        dbp = np.array([[80.0], [82.0], [85.0], [88.0], [90.0], [92.0]], dtype=np.float32)
        age = np.array([45.0, 50.0, np.nan, 60.0, 65.0, 70.0], dtype=np.float32)
        bmi = np.array([22.5, np.nan, 24.0, 26.5, 28.0, 25.0], dtype=np.float32)
        subjects = ["p000101", "p000101", "p000201", "p000201", "p000301", "p000301"]

        subset_dict = {
            "Signals": signals,
            "SBP": sbp,
            "DBP": dbp,
            "Age": age,
            "BMI": bmi,
            "Subject": subjects
        }

        dataset = PulseDBDataset.from_subset_dict(subset_dict, target_fs=30.0, orig_fs=125.0)
        assert len(dataset) == 6

        item = dataset[0]
        # Resampled 1250 samples @ 125 Hz -> 300 samples @ 30 Hz
        assert item["morph"].shape == (1, 300)
        assert item["timing"].shape == (1, 300)
        assert item["anchor"].shape == (6,)
        assert item["sbp"].item() == 120.0
        assert item["dbp"].item() == 80.0
        assert item["subject_id"] == "p000101"

        # Check NaN imputation in anchors
        item_nan_age = dataset[2]
        assert not np.isnan(item_nan_age["anchor"].numpy()).any()
        item_nan_bmi = dataset[1]
        assert not np.isnan(item_nan_bmi["anchor"].numpy()).any()

        # Batch collation and model forward pass test
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_bp_batch)
        batch = next(iter(loader))
        assert batch["morph"].shape == (2, 1, 300)
        assert batch["timing"].shape == (2, 1, 300)
        assert batch["anchor"].shape == (2, 6)

        model = DualStreamTCN()
        sbp_pred, dbp_pred = model(batch["morph"], batch["timing"], batch["anchor"])
        assert sbp_pred.shape == (2, 1)
        assert dbp_pred.shape == (2, 1)

    def test_subject_independent_split_strict_separation(self):
        """subject_independent_split must produce partitions with zero subject overlap."""
        # 4 unique subjects, 4 segments each
        subjects = ["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4 + ["s4"] * 4
        n_total = len(subjects)
        ds = DualSiteBPDataset(
            morph_signals=np.zeros((n_total, 1, 50)),
            timing_signals=np.zeros((n_total, 1, 50)),
            anchor_vectors=np.zeros((n_total, 6)),
            sbp_targets=np.zeros((n_total, 1)),
            dbp_targets=np.zeros((n_total, 1)),
            subject_ids=subjects
        )

        train_set, val_set, test_set = subject_independent_split(ds, 0.50, 0.25, 0.25, seed=42)

        train_subjects = {ds.subject_ids[i] for i in train_set.indices}
        val_subjects = {ds.subject_ids[i] for i in val_set.indices}
        test_subjects = {ds.subject_ids[i] for i in test_set.indices}

        # Sets must be mutually disjoint
        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)
        assert train_subjects.union(val_subjects).union(test_subjects) == {"s1", "s2", "s3", "s4"}

    def test_pulsedb_from_mat_file_not_found(self):
        """Missing MAT file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PulseDBDataset.from_mat_file("non_existent_file.mat")



# ============================================================================
# 4. Trainer and Convergence Tests
# ============================================================================
class TestTrainerAndConvergence:
    """Verifies trainer optimization steps, validation loop, and checkpointing."""

    def test_trainer_single_epoch_execution(self):
        """Trainer must execute one training epoch and return valid metrics."""
        model = DualStreamTCN()
        ds = generate_synthetic_dataset(num_samples=16, seq_len=100, seed=1)
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_bp_batch)

        trainer = BPTrainer(model, config=TrainingConfig(epochs=1, device="cpu"))
        train_res = trainer.train_epoch(loader)

        assert "train_loss" in train_res
        assert train_res["train_loss"] > 0.0

    def test_trainer_evaluation_metrics(self):
        """Evaluation loop must compute MAE, RMSE, Pearson r, and AAMI status."""
        model = DualStreamTCN()
        ds = generate_synthetic_dataset(num_samples=20, seq_len=100, seed=2)
        loader = DataLoader(ds, batch_size=10, collate_fn=collate_bp_batch)

        trainer = BPTrainer(model, config=TrainingConfig(device="cpu"))
        metrics = trainer.evaluate(loader)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.sbp_mae >= 0.0
        assert metrics.dbp_mae >= 0.0
        assert -1.0 <= metrics.sbp_pearson_r <= 1.0
        assert metrics.sbp_variance_ratio >= 0.0

    def test_trainer_fit_and_checkpoint_roundtrip(self):
        """fit() must run multi-epoch training and restore/save checkpoints."""
        model = DualStreamTCN()
        train_ds = generate_synthetic_dataset(num_samples=16, seq_len=100, seed=3)
        val_ds = generate_synthetic_dataset(num_samples=8, seq_len=100, seed=4)

        train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collate_bp_batch)
        val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_bp_batch)

        trainer = BPTrainer(model, config=TrainingConfig(epochs=2, patience=2, device="cpu"))
        history = trainer.fit(train_loader, val_loader, epochs=2)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model_test.pt")
            trainer.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            new_model = DualStreamTCN()
            new_trainer = BPTrainer(new_model)
            new_trainer.load_checkpoint(ckpt_path)

            # Compare weights
            for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
                assert torch.allclose(p1, p2)


# ============================================================================
# 5. INT8 Quantization and Edge Profiling Tests
# ============================================================================
class TestQuantizerAndEdgeProfiling:
    """Verifies INT8 quantization math, SNR fidelity, mobile profiler, and file exports."""

    def test_symmetric_weight_quantization_properties(self):
        """Symmetric quantization must constrain weights to [-127, 127] with scale > 0."""
        w = np.random.normal(0.0, 0.5, size=(64, 64)).astype(np.float32)
        q_w, scale = quantize_symmetric(w)

        assert q_w.dtype == np.int8
        assert scale > 0.0
        assert np.all(q_w >= -127) and np.all(q_w <= 127)

    def test_asymmetric_activation_quantization(self):
        """Asymmetric quantization must produce INT8 values in [-128, 127] with zero-point."""
        acts = np.linspace(-3.0, 5.0, 200).astype(np.float32)
        q_acts, scale, zp = quantize_asymmetric(acts)

        assert q_acts.dtype == np.int8
        assert scale > 0.0
        assert -128 <= zp <= 127
        assert np.all(q_acts >= -128) and np.all(q_acts <= 127)

        dequant = dequantize_asymmetric(q_acts, scale, zp)
        max_err = float(np.max(np.abs(acts - dequant)))
        assert max_err < 0.1

    def test_quantization_snr_preservation(self):
        """INT8 symmetric quantization must preserve SNR >= 35 dB."""
        w = np.random.normal(0.0, 1.0, size=(128, 128)).astype(np.float32)
        q_w, scale = quantize_symmetric(w)
        dequant = dequantize_symmetric(q_w, scale)
        snr = calculate_quantization_error_snr(w, dequant)
        assert snr >= 35.0, f"Quantization SNR {snr:.2f} dB is below 35 dB target"

    def test_edge_device_profiling_limits(self):
        """Model profiler must verify Snapdragon 4 Gen 2 budget (<60ms, <=20MB, <250KB)."""
        quantizer = INT8Quantizer()
        profile = quantizer.profile_model(input_seq_len=300)

        assert profile["total_parameters"] < 250_000
        assert profile["compression_ratio"] == 4.0
        assert profile["meets_latency_budget"] is True
        assert profile["meets_memory_budget"] is True
        assert profile["meets_size_budget"] is True
        assert profile["cpu_latency_ms"] < 60.0
        assert profile["peak_memory_mb"] <= 20.0
        assert profile["int8_size_kb"] < 250.0

    def test_export_onnx_artifact(self):
        """export_onnx must generate valid file artifact."""
        quantizer = INT8Quantizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.onnx")
            res = quantizer.export_onnx(save_path, seq_len=100)
            assert os.path.exists(save_path)
            assert res["file_size_kb"] > 0.0

    def test_export_tflite_int8_flatbuffer(self):
        """export_tflite_int8 must generate flatbuffer under 250 KB with 'TFL3' magic header."""
        quantizer = INT8Quantizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "dual_stream_tcn_int8.tflite")
            res = quantizer.export_tflite_int8(save_path)

            assert os.path.exists(save_path)
            assert res["file_size_kb"] < 250.0
            assert res["average_snr_db"] >= 35.0
            assert res["meets_size_target"] is True

            # Verify magic header bytes
            with open(save_path, "rb") as f:
                header = f.read(8)
                assert b"TFL3" in header

            # Verify loading and deserialization
            tensors = INT8Quantizer.load_tflite_int8(save_path)
            assert len(tensors) > 0
            # Inspect first quantized weight tensor
            for name, tensor_info in tensors.items():
                if "q_weights" in tensor_info:
                    assert tensor_info["q_weights"].dtype == np.int8
                    assert tensor_info["scale"] > 0.0
                    break

    def test_fake_quantize_ste_gradient_propagation(self):
        """FakeQuantizeWeight module must allow unattenuated backprop in QAT."""
        fake_q = FakeQuantizeWeight()
        w = torch.randn(10, 10, requires_grad=True)
        out = fake_q(w)
        loss = out.sum()
        loss.backward()

        assert w.grad is not None
        # STE should pass gradients through
        assert not torch.allclose(w.grad, torch.zeros_like(w.grad))
