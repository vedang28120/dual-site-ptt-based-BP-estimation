"""Physics-Informed Hemodynamic Engine (HRVP-PINN).

This package provides analytical hemodynamic parameter identification,
sub-frame intersecting tangent foot detection, anthropometric path modeling,
IMU hydrostatic perturbation calculations, Traube-Hering-Mayer envelope extraction,
and closed-form arterial stiffness parameter solving.
"""

from .anthropometrics import (
    Anthropometrics,
    compute_epwv,
    compute_epwv_from_profile,
    compute_path_length,
)
from .foot_detector import (
    FootDetector,
    calculate_tangent_foot,
    compute_subframe_ptt,
    detect_pulse_feet,
)
from .hydrostatic_engine import (
    G_GRAVITY,
    MMHG_PER_METER,
    MMHG_PER_MM,
    PA_TO_MMHG,
    RHO_BLOOD,
    HydrostaticEngine,
    elevation_to_hydrostatic_pressure,
    integrate_zupt_elevation,
    pressure_to_elevation,
)
from .respiratory_envelope import (
    RespiratoryEnvelopeExtractor,
    compute_compliance_slope,
    extract_respiratory_envelope,
)
from .stiffness_solver import (
    HRVPEngine,
    calculate_k_form,
    construct_anchor_vector,
    decompose_sbp_dbp,
    solve_hrvp,
    solve_stiffness_parameters,
)
from .pulse_overlap_verifier import (
    ExactPulseRegistrationEngine,
    compute_povi,
    compute_mda,
    compute_upstroke_correlation,
)
from .anthropometrics import (
    OrthogonalPhysiologicalExtractor,
)

__all__ = [
    # Foot detector
    "calculate_tangent_foot",
    "detect_pulse_feet",
    "compute_subframe_ptt",
    "FootDetector",
    # Registration & Verification
    "ExactPulseRegistrationEngine",
    "compute_povi",
    "compute_mda",
    "compute_upstroke_correlation",
    # Anthropometrics
    "compute_path_length",
    "compute_epwv",
    "compute_epwv_from_profile",
    "Anthropometrics",
    "OrthogonalPhysiologicalExtractor",
    # Hydrostatic
    "elevation_to_hydrostatic_pressure",
    "pressure_to_elevation",
    "integrate_zupt_elevation",
    "HydrostaticEngine",
    "RHO_BLOOD",
    "G_GRAVITY",
    "PA_TO_MMHG",
    "MMHG_PER_METER",
    "MMHG_PER_MM",
    # Respiratory envelope
    "compute_compliance_slope",
    "extract_respiratory_envelope",
    "RespiratoryEnvelopeExtractor",
    # Stiffness solver
    "calculate_k_form",
    "solve_stiffness_parameters",
    "decompose_sbp_dbp",
    "construct_anchor_vector",
    "solve_hrvp",
    "HRVPEngine",
]
