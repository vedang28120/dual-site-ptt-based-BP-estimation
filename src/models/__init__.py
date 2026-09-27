"""
src/models: Anti-Collapse Edge Deep Learning Architecture DS-TCN & INT8 Quantization.

Exposes:
- Neural Network: DualStreamTCN, CausalConv1d, DilatedCausalBlock, CausalCrossAttention
- Compound Loss: TripartiteCompoundLoss, PyTorchTripartiteLoss, NumpyTripartiteLoss, ReferenceTripartiteLoss
- Dataset: DualSiteBPDataset, collate_bp_batch, generate_synthetic_dataset, train_val_test_split
- Training & Eval: BPTrainer, TrainingConfig, EvaluationMetrics
- Quantization & Edge: INT8Quantizer, ReferenceINT8Quantizer, quantize_symmetric, dequantize_symmetric,
                       quantize_asymmetric, dequantize_asymmetric, calculate_quantization_error_snr
"""

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
)

__all__ = [
    # DS-TCN
    "DualStreamTCN",
    "CausalConv1d",
    "DilatedCausalBlock",
    "CausalCrossAttention",
    # Compound Loss
    "TripartiteCompoundLoss",
    "PyTorchTripartiteLoss",
    "NumpyTripartiteLoss",
    "ReferenceTripartiteLoss",
    # Dataset
    "DualSiteBPDataset",
    "PulseDBDataset",
    "collate_bp_batch",
    "generate_synthetic_dataset",
    "train_val_test_split",
    "subject_independent_split",
    # Trainer
    "BPTrainer",
    "TrainingConfig",
    "EvaluationMetrics",
    # Quantizer
    "INT8Quantizer",
    "ReferenceINT8Quantizer",
    "quantize_symmetric",
    "dequantize_symmetric",
    "quantize_asymmetric",
    "dequantize_asymmetric",
    "calculate_quantization_error_snr",
]
