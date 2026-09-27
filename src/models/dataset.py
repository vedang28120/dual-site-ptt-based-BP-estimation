"""
PyTorch Dataset and Data Generation Utilities for Dual-Site BP Estimation.

Manages synchronized dual-stream optical sequences (contact morphology + facial timing),
6D HRVP physiological anchor vectors, and clinical SBP/DBP targets.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from scipy.signal import resample
from scipy.io import loadmat


class DualSiteBPDataset(Dataset):
    """
    PyTorch Dataset wrapping synchronized dual-site optical signals and HRVP anchors.
    
    Item dictionary contains:
      - 'morph': [1, T] float32 tensor (contact finger PPG morphology)
      - 'timing': [1, T] float32 tensor (facial rPPG macro-foot timing)
      - 'anchor': [6] float32 tensor (HRVP physiological anchor)
      - 'sbp': [1] float32 tensor (Systolic Blood Pressure target)
      - 'dbp': [1] float32 tensor (Diastolic Blood Pressure target)
      - 'targets': [2] float32 tensor ([SBP, DBP])
      - 'subject_id': str or int identifier
    """
    def __init__(
        self,
        morph_signals: Union[np.ndarray, torch.Tensor],
        timing_signals: Union[np.ndarray, torch.Tensor],
        anchor_vectors: Union[np.ndarray, torch.Tensor],
        sbp_targets: Union[np.ndarray, torch.Tensor],
        dbp_targets: Union[np.ndarray, torch.Tensor],
        subject_ids: Optional[Union[np.ndarray, List[Any]]] = None
    ) -> None:
        super().__init__()

        # Convert to torch.float32 tensors
        self.morph = torch.as_tensor(morph_signals, dtype=torch.float32)
        self.timing = torch.as_tensor(timing_signals, dtype=torch.float32)
        self.anchor = torch.as_tensor(anchor_vectors, dtype=torch.float32)
        self.sbp = torch.as_tensor(sbp_targets, dtype=torch.float32)
        self.dbp = torch.as_tensor(dbp_targets, dtype=torch.float32)

        # Standardize channel dimensions to [N, 1, T]
        if self.morph.ndim == 2:
            self.morph = self.morph.unsqueeze(1)
        if self.timing.ndim == 2:
            self.timing = self.timing.unsqueeze(1)
        if self.sbp.ndim == 1:
            self.sbp = self.sbp.unsqueeze(1)
        if self.dbp.ndim == 1:
            self.dbp = self.dbp.unsqueeze(1)

        self.num_samples = self.morph.shape[0]
        if not (len(self.timing) == len(self.anchor) == len(self.sbp) == len(self.dbp) == self.num_samples):
            raise ValueError(
                f"Dimension mismatch across inputs: morph={self.num_samples}, timing={len(self.timing)}, "
                f"anchor={len(self.anchor)}, sbp={len(self.sbp)}, dbp={len(self.dbp)}"
            )

        if subject_ids is not None:
            self.subject_ids = list(subject_ids)
        else:
            self.subject_ids = [f"sub_{i:04d}" for i in range(self.num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "morph": self.morph[idx],
            "timing": self.timing[idx],
            "anchor": self.anchor[idx],
            "sbp": self.sbp[idx],
            "dbp": self.dbp[idx],
            "targets": torch.cat([self.sbp[idx], self.dbp[idx]], dim=-1),
            "subject_id": self.subject_ids[idx]
        }


def collate_bp_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader batching.
    
    Stacks tensors and aggregates subject metadata.
    """
    morph = torch.stack([item["morph"] for item in batch], dim=0)
    timing = torch.stack([item["timing"] for item in batch], dim=0)
    anchor = torch.stack([item["anchor"] for item in batch], dim=0)
    sbp = torch.stack([item["sbp"] for item in batch], dim=0)
    dbp = torch.stack([item["dbp"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    subject_ids = [item["subject_id"] for item in batch]

    return {
        "morph": morph,
        "timing": timing,
        "anchor": anchor,
        "sbp": sbp,
        "dbp": dbp,
        "targets": targets,
        "subject_ids": subject_ids
    }


def generate_synthetic_dataset(
    num_samples: int = 100,
    seq_len: int = 300,
    seed: int = 42
) -> DualSiteBPDataset:
    """
    Generates a realistic synthetic dual-site physiological dataset.
    
    Simulates:
      - Subject arterial stiffness parameters (E0, alpha)
      - Sub-frame transit delay delta PTT
      - Contact PPG morphology with dicrotic notch
      - Facial rPPG with early arrival timing
      - Hydrostatic perturbation delta P_hydro
      - Ground truth SBP and DBP following AAMI/ISO distributions
    """
    rng = np.random.RandomState(seed)

    morph_list = []
    timing_list = []
    anchor_list = []
    sbp_list = []
    dbp_list = []
    sub_ids = []

    t_sec = np.linspace(0, seq_len / 30.0, seq_len, endpoint=False)

    for i in range(num_samples):
        # 1. Hemodynamic profile
        sbp = float(rng.normal(122.0, 16.0))
        sbp = float(np.clip(sbp, 85.0, 185.0))
        dbp = float(sbp - rng.uniform(32.0, 55.0))
        dbp = float(np.clip(dbp, 50.0, 115.0))
        map_val = dbp + (1.0 / 3.0) * (sbp - dbp)

        hr_bpm = float(rng.uniform(55.0, 95.0))
        hr_hz = hr_bpm / 60.0

        # Physical stiffness parameters
        alpha = float(rng.uniform(0.018, 0.032))
        e0 = float(rng.uniform(70000.0, 95000.0))
        c0 = float(np.sqrt(e0 * 0.001 / (2.0 * 1055.0 * 0.0015)))  # ~5.0 m/s
        epwv = float(c0 * np.exp(0.5 * alpha * map_val))  # ~6.0 - 9.0 m/s
        d_path = float(rng.uniform(1.10, 1.25))
        delta_ptt_sec = d_path / epwv
        delta_ptt_ms = float(delta_ptt_sec * 1000.0)

        delta_h = float(rng.uniform(-0.30, 0.30))
        delta_p_hydro = float(77.6014 * delta_h)
        compliance_slope = float((alpha / 2.0) * epwv)

        anchor = np.array([
            epwv,
            delta_ptt_ms,
            delta_p_hydro,
            compliance_slope,
            e0,
            alpha
        ], dtype=np.float32)

        # 2. Waveforms
        # Contact Finger PPG (rich morphology: fundamental + dicrotic notch harmonic)
        phase_contact = 2.0 * np.pi * hr_hz * t_sec
        systolic_component = -np.cos(phase_contact)
        dicrotic_notch = 0.35 * np.cos(2.0 * phase_contact - 0.7)
        noise_finger = rng.normal(0.0, 0.02, size=seq_len)
        finger_wave = systolic_component + dicrotic_notch + noise_finger
        # Normalize to [0, 1]
        finger_wave = (finger_wave - np.min(finger_wave)) / (np.max(finger_wave) - np.min(finger_wave) + 1e-6)

        # Facial rPPG (arrives earlier by delta_ptt_sec)
        phase_face = 2.0 * np.pi * hr_hz * (t_sec + delta_ptt_sec)
        face_wave = -np.cos(phase_face) + 0.15 * np.cos(2.0 * phase_face - 0.7)
        noise_face = rng.normal(0.0, 0.06, size=seq_len)
        face_wave = face_wave + noise_face
        face_wave = (face_wave - np.min(face_wave)) / (np.max(face_wave) - np.min(face_wave) + 1e-6)

        morph_list.append(finger_wave.reshape(1, seq_len))
        timing_list.append(face_wave.reshape(1, seq_len))
        anchor_list.append(anchor)
        sbp_list.append([sbp])
        dbp_list.append([dbp])
        sub_ids.append(f"subject_{i // 5:03d}")

    return DualSiteBPDataset(
        morph_signals=np.array(morph_list, dtype=np.float32),
        timing_signals=np.array(timing_list, dtype=np.float32),
        anchor_vectors=np.array(anchor_list, dtype=np.float32),
        sbp_targets=np.array(sbp_list, dtype=np.float32),
        dbp_targets=np.array(dbp_list, dtype=np.float32),
        subject_ids=sub_ids
    )


def train_val_test_split(
    dataset: DualSiteBPDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split DualSiteBPDataset into train, validation, and test subsets.
    """
    total = len(dataset)
    indices = np.arange(total)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx)
    )


def subject_independent_split(
    dataset: DualSiteBPDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Splits DualSiteBPDataset into subject-independent train, validation, and test subsets.
    
    Guarantees no subject appears across multiple partitions, strictly preventing data
    leakage during calibration-free benchmarking.
    """
    subject_to_indices: Dict[Any, List[int]] = {}
    for idx, sid in enumerate(dataset.subject_ids):
        subject_to_indices.setdefault(sid, []).append(idx)

    unique_subjects = list(subject_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)

    n_subj = len(unique_subjects)
    n_train_sub = max(1, int(round(n_subj * train_ratio)))
    n_val_sub = max(1, int(round(n_subj * val_ratio))) if (val_ratio > 0 and n_subj > 2) else 0

    train_subs = set(unique_subjects[:n_train_sub])
    val_subs = set(unique_subjects[n_train_sub:n_train_sub + n_val_sub])
    test_subs = set(unique_subjects[n_train_sub + n_val_sub:])

    train_idx = [idx for s in train_subs for idx in subject_to_indices[s]]
    val_idx = [idx for s in val_subs for idx in subject_to_indices[s]]
    test_idx = [idx for s in test_subs for idx in subject_to_indices[s]]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx)
    )


class PulseDBDataset(DualSiteBPDataset):
    """
    Dataset loader for PulseDB benchmark data (Frontiers in Digital Health, 2023).
    
    PulseDB consists of 5,245,454 segments from 5,361 subjects across MIMIC-III and VitalDB.
    This loader processes:
      1. Predefined subset matrices generated by Generate_Subsets.m:
         - Subset.Signals: [N, 3, 1250] (Channel 0: ECG_F, Channel 1: PPG_F, Channel 2: ABP_Raw)
         - Subset.SBP: [N, 1] SegSBP labels in mmHg
         - Subset.DBP: [N, 1] SegDBP labels in mmHg
         - Demographics: Age, Gender, Height, Weight, BMI
      2. Resampling from native 125 Hz (1250 samples / 10s) to target edge rate (30 Hz / 300 samples).
      3. Synthesis of 6D HRVP physiological anchors:
         [ePWV_proxy, delta_ptt_ms, delta_p_hydro, compliance_slope, age, bmi]
      4. Strict subject-independent evaluation splits to prevent inter-subject data leakage.
    """
    def __init__(
        self,
        morph_signals: Union[np.ndarray, torch.Tensor],
        timing_signals: Union[np.ndarray, torch.Tensor],
        anchor_vectors: Union[np.ndarray, torch.Tensor],
        sbp_targets: Union[np.ndarray, torch.Tensor],
        dbp_targets: Union[np.ndarray, torch.Tensor],
        subject_ids: Optional[Union[np.ndarray, List[Any]]] = None,
        source_name: str = "PulseDB"
    ) -> None:
        super().__init__(
            morph_signals=morph_signals,
            timing_signals=timing_signals,
            anchor_vectors=anchor_vectors,
            sbp_targets=sbp_targets,
            dbp_targets=dbp_targets,
            subject_ids=subject_ids
        )
        self.source_name = source_name

    @classmethod
    def from_subset_dict(
        cls,
        subset_dict: Dict[str, Any],
        target_fs: float = 30.0,
        orig_fs: float = 125.0,
        max_samples: Optional[int] = None
    ) -> "PulseDBDataset":
        """
        Constructs a PulseDBDataset from an in-memory dictionary matching PulseDB schema.
        
        Args:
            subset_dict: Dict with 'Signals', 'SBP', 'DBP', and optionally 'Subject', 'Age', 'BMI'.
            target_fs: Desired sample rate for DS-TCN model (default 30 Hz).
            orig_fs: Native PulseDB sample rate (default 125 Hz).
            max_samples: Optional limit on the number of samples loaded.
        """
        signals = subset_dict["Signals"]  # [N, 3, 1250] or [N, 2, 1250]
        sbp = np.asarray(subset_dict["SBP"], dtype=np.float32).reshape(-1, 1)
        dbp = np.asarray(subset_dict["DBP"], dtype=np.float32).reshape(-1, 1)

        n_total = signals.shape[0]
        if max_samples is not None:
            n_total = min(n_total, max_samples)
            signals = signals[:n_total]
            sbp = sbp[:n_total]
            dbp = dbp[:n_total]

        # Resample signals if target_fs != orig_fs
        n_orig_samples = signals.shape[-1]
        target_len = int(round(n_orig_samples * (target_fs / orig_fs)))
        
        # Channel 1: PPG_F (Finger Morphology), Channel 0: ECG_F (Timing reference)
        if signals.ndim == 3 and signals.shape[1] >= 2:
            ecg_raw = signals[:, 0, :]
            ppg_raw = signals[:, 1, :]
        elif signals.ndim == 3 and signals.shape[1] == 1:
            ppg_raw = signals[:, 0, :]
            ecg_raw = signals[:, 0, :]
        elif signals.ndim == 2:
            ppg_raw = signals
            ecg_raw = signals
        else:
            raise ValueError(f"Unexpected signals shape: {signals.shape}")

        if target_len != n_orig_samples:
            # Resample along time axis
            ppg_resampled = resample(ppg_raw, target_len, axis=-1).astype(np.float32)
            ecg_resampled = resample(ecg_raw, target_len, axis=-1).astype(np.float32)
        else:
            ppg_resampled = ppg_raw.astype(np.float32)
            ecg_resampled = ecg_raw.astype(np.float32)

        # Min-max normalize per segment to ensure numerical stability [0, 1]
        ppg_min = np.min(ppg_resampled, axis=-1, keepdims=True)
        ppg_max = np.max(ppg_resampled, axis=-1, keepdims=True)
        ppg_norm = (ppg_resampled - ppg_min) / (ppg_max - ppg_min + 1e-6)

        ecg_min = np.min(ecg_resampled, axis=-1, keepdims=True)
        ecg_max = np.max(ecg_resampled, axis=-1, keepdims=True)
        ecg_norm = (ecg_resampled - ecg_min) / (ecg_max - ecg_min + 1e-6)

        # Demographics & Anchors
        age_arr = subset_dict.get("Age")
        if age_arr is not None:
            age_arr = np.asarray(age_arr, dtype=np.float32).reshape(-1)
            if len(age_arr) > n_total:
                age_arr = age_arr[:n_total]
            nan_mask = np.isnan(age_arr)
            if np.any(nan_mask):
                valid_median = float(np.nanmedian(age_arr)) if not np.all(nan_mask) else 55.0
                age_arr[nan_mask] = valid_median
        else:
            age_arr = np.full(n_total, 55.0, dtype=np.float32)

        bmi_arr = subset_dict.get("BMI")
        if bmi_arr is not None:
            bmi_arr = np.asarray(bmi_arr, dtype=np.float32).reshape(-1)
            if len(bmi_arr) > n_total:
                bmi_arr = bmi_arr[:n_total]
            nan_mask = np.isnan(bmi_arr)
            if np.any(nan_mask):
                valid_median = float(np.nanmedian(bmi_arr)) if not np.all(nan_mask) else 25.0
                bmi_arr[nan_mask] = valid_median
        else:
            bmi_arr = np.full(n_total, 25.0, dtype=np.float32)

        # Construct 6D HRVP Anchors:
        # [ePWV_proxy, delta_ptt_ms, delta_p_hydro, compliance_slope, age, bmi]
        epwv_proxy = 5.0 + 0.05 * age_arr
        delta_ptt_ms = (1.2 / np.maximum(epwv_proxy, 1e-3)) * 1000.0
        delta_p_hydro = np.zeros(n_total, dtype=np.float32)  # Supine ICU baseline
        compliance_slope = 0.02 * epwv_proxy

        anchors = np.column_stack([
            epwv_proxy,
            delta_ptt_ms,
            delta_p_hydro,
            compliance_slope,
            age_arr,
            bmi_arr
        ]).astype(np.float32)

        # Subject IDs
        subj_list = subset_dict.get("Subject")
        if subj_list is not None:
            if isinstance(subj_list, np.ndarray):
                subj_list = [str(s.item() if hasattr(s, "item") else s) for s in subj_list[:n_total]]
            else:
                subj_list = [str(s) for s in list(subj_list)[:n_total]]
        else:
            subj_list = [f"pulsedb_{i:05d}" for i in range(n_total)]

        return cls(
            morph_signals=ppg_norm,
            timing_signals=ecg_norm,
            anchor_vectors=anchors,
            sbp_targets=sbp,
            dbp_targets=dbp,
            subject_ids=subj_list,
            source_name="PulseDB"
        )

    @classmethod
    def from_mat_file(
        cls,
        mat_path: str,
        target_fs: float = 30.0,
        orig_fs: float = 125.0,
        max_samples: Optional[int] = None
    ) -> "PulseDBDataset":
        """
        Loads a PulseDB MAT file (MAT v5/v7 or v7.3 HDF5) and creates a PulseDBDataset.
        """
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"PulseDB file not found at: {mat_path}")

        try:
            # Attempt loadmat (v5/v7 up to 7.2)
            mat_contents = loadmat(mat_path, squeeze_me=True)
            if "Subset" in mat_contents:
                raw_subset = mat_contents["Subset"]
                subset_dict = {
                    name: raw_subset[name].item() if hasattr(raw_subset[name], "item") else raw_subset[name]
                    for name in raw_subset.dtype.names
                }
            else:
                subset_dict = mat_contents
            return cls.from_subset_dict(subset_dict, target_fs=target_fs, orig_fs=orig_fs, max_samples=max_samples)
        except NotImplementedError:
            # File is MAT-File v7.3 (HDF5 based)
            try:
                import mat73
                data = mat73.loadmat(mat_path)
                subset_data = data.get("Subset", data)
                return cls.from_subset_dict(subset_data, target_fs=target_fs, orig_fs=orig_fs, max_samples=max_samples)
            except ImportError:
                try:
                    import h5py
                    with h5py.File(mat_path, "r") as f:
                        root = f["Subset"] if "Subset" in f else f
                        signals = np.array(root["Signals"])
                        sbp = np.array(root["SBP"])
                        dbp = np.array(root["DBP"])
                        subset_dict = {"Signals": signals, "SBP": sbp, "DBP": dbp}
                        return cls.from_subset_dict(subset_dict, target_fs=target_fs, orig_fs=orig_fs, max_samples=max_samples)
                except ImportError:
                    raise ImportError(
                        f"Failed to load MAT v7.3 file '{mat_path}'. "
                        "Please install 'mat73' or 'h5py' via: pip install mat73 h5py"
                    )
