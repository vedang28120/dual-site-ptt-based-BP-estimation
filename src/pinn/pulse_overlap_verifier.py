"""
Exact Pulse-to-Pulse Overlapping & Registration Engine with Tripartite Certification Gate.

Replaces blind cross-correlation with physiological beat-by-beat dynamic registration:
  1. Sub-sample parabolic Vmax and Tangent-Intersection Foot solver:
       t_foot = t_vmax - (y(t_vmax) - y_min) / v_max
  2. Physiological transit time gating: tau in [30 ms, 220 ms]
  3. APG (a, b, c, d, e) 2nd-derivative wave fiducial extractor
  4. Tripartite certification gate:
       - Pulse Overlap Validity Index (POVI >= 0.90)
       - Morphological Discrepancy Area (MDA <= 0.12)
       - Systolic upstroke Pearson correlation (r_upstroke >= 0.95)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp


def compute_povi(face_beat_norm: np.ndarray, palm_beat_norm: np.ndarray) -> float:
    """Computes Pulse Overlap Validity Index (POVI) between normalized [0, 1] single-beat waveforms."""
    dot_prod = np.dot(face_beat_norm, palm_beat_norm)
    norm_product = np.linalg.norm(face_beat_norm) * np.linalg.norm(palm_beat_norm) + 1e-8
    return float(dot_prod / norm_product)


def compute_mda(face_beat_norm: np.ndarray, palm_beat_norm: np.ndarray) -> float:
    """Computes Morphological Discrepancy Area (MDA) between normalized [0, 1] single-beat waveforms."""
    return float(np.mean(np.abs(face_beat_norm - palm_beat_norm)))


def compute_upstroke_correlation(
    face_beat_norm: np.ndarray, palm_beat_norm: np.ndarray, upstroke_len: int
) -> float:
    """Computes Pearson correlation coefficient over the systolic upstroke segment."""
    up_len = max(5, min(upstroke_len, len(face_beat_norm), len(palm_beat_norm)))
    f_up = face_beat_norm[:up_len]
    p_up = palm_beat_norm[:up_len]
    if np.std(f_up) < 1e-6 or np.std(p_up) < 1e-6:
        return 0.0
    r = np.corrcoef(f_up, p_up)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


class ExactPulseRegistrationEngine:
    """
    Beat-by-beat dynamic registration engine featuring:
      - Sub-sample parabolic Vmax and Tangent-Intersection Foot solver
      - Physiological transit time gating: tau in [30 ms, 220 ms]
      - APG (a, b, c, d, e) 2nd-derivative wave extractor
      - Tripartite certification gate: POVI >= 0.90, MDA <= 0.12, r_upstroke >= 0.95
    """

    def __init__(self, fs: int = 250, min_ptt_ms: float = 30.0, max_ptt_ms: float = 220.0):
        self.fs = fs
        self.min_ptt_sec = min_ptt_ms / 1000.0
        self.max_ptt_sec = max_ptt_ms / 1000.0

    def solve_tangent_foot(
        self, ppg: np.ndarray, vpg: np.ndarray, v_peak_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Solves sub-sample t_foot = t_vmax - (y(t_vmax) - y_min) / v_max.
        Refines v_max and t_vmax via a 3-point parabolic interpolation.
        """
        N = len(ppg)
        if v_peak_idx <= 2 or v_peak_idx >= N - 2:
            return None

        # 3-point parabolic refinement of Vmax
        v_m1, v_0, v_p1 = vpg[v_peak_idx - 1], vpg[v_peak_idx], vpg[v_peak_idx + 1]
        denom = v_m1 - 2.0 * v_0 + v_p1
        delta = 0.5 * (v_m1 - v_p1) / denom if abs(denom) > 1e-9 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))

        t_vmax = (v_peak_idx + delta) / self.fs
        v_max = float(v_0 - 0.25 * (v_m1 - v_p1) * delta)
        if v_max <= 1e-6:
            return None

        # Interpolate y(t_vmax)
        y_vmax = float(ppg[v_peak_idx] + delta * (ppg[v_peak_idx + 1] - ppg[v_peak_idx - 1]) * 0.5)

        # Preceding diastolic minimum within [-160 ms, 0 ms]
        search_start = max(0, v_peak_idx - int(0.16 * self.fs))
        min_rel_idx = int(np.argmin(ppg[search_start : v_peak_idx + 1]))
        min_idx = search_start + min_rel_idx
        y_min = float(ppg[min_idx])

        t_foot = t_vmax - (y_vmax - y_min) / v_max
        if not (search_start / self.fs <= t_foot <= t_vmax):
            t_foot = min_idx / self.fs

        # Locate systolic peak within [t_vmax, t_vmax + 220 ms]
        sys_end = min(N, v_peak_idx + int(0.22 * self.fs))
        sys_idx = v_peak_idx + int(np.argmax(ppg[v_peak_idx:sys_end]))
        y_sys = float(ppg[sys_idx])
        t_sys = sys_idx / self.fs

        return {
            "t_foot": float(t_foot),
            "t_vmax": float(t_vmax),
            "v_max": v_max,
            "t_sys": t_sys,
            "y_min": y_min,
            "y_sys": y_sys,
            "foot_idx": int(round(t_foot * self.fs)),
            "sys_idx": sys_idx,
        }

    def extract_apg_waves(self, apg_beat: np.ndarray, v_max_rel_idx: int) -> Dict[str, Any]:
        """Extracts canonical 2nd derivative APG (a, b, c, d, e) amplitudes and indices."""
        L = len(apg_beat)
        if L < 5:
            return {"a": 1.0, "b": -0.5, "c": 0.2, "d": -0.2, "e": 0.3, "e_idx": max(1, L - 1)}

        a_search_end = min(L, v_max_rel_idx + max(2, int(0.02 * self.fs)))
        a_idx = int(np.argmax(apg_beat[: max(a_search_end, 3)]))
        a_val = float(max(apg_beat[a_idx], 1e-6))

        b_end = min(L, a_idx + int(0.14 * self.fs))
        b_idx = a_idx + int(np.argmin(apg_beat[a_idx : max(b_end, a_idx + 2)]))
        b_val = float(apg_beat[b_idx])

        e_start = min(L - 3, b_idx + int(0.06 * self.fs))
        e_end = min(L, a_idx + int(0.45 * self.fs))
        if e_end > e_start + 2:
            e_idx = e_start + int(np.argmax(apg_beat[e_start:e_end]))
        else:
            e_idx = min(L - 1, b_idx + 4)
        e_val = float(apg_beat[e_idx])

        if e_idx > b_idx + 2:
            sub_cd = apg_beat[b_idx + 1 : e_idx]
            c_idx = b_idx + 1 + int(np.argmax(sub_cd))
            c_val = float(apg_beat[c_idx])
            if e_idx > c_idx + 1:
                d_idx = c_idx + int(np.argmin(apg_beat[c_idx:e_idx]))
                d_val = float(apg_beat[d_idx])
            else:
                d_val = c_val
        else:
            c_val = 0.5 * (b_val + e_val)
            d_val = c_val

        return {"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val, "e_idx": e_idx}

    def register_and_verify_pulses(
        self, face_dict: Dict[str, np.ndarray], palm_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Matches Face (proximal) and Palm (distal) beats within [30 ms, 220 ms],
        computes sub-sample PTT_foot and PTT_vmax, and certifies overlap via POVI & MDA.
        """
        f_ppg, f_vpg, f_apg = face_dict["ppg"], face_dict["vpg"], face_dict["apg"]
        p_ppg, p_vpg, p_apg = palm_dict["ppg"], palm_dict["vpg"], palm_dict["apg"]

        min_dist = int(0.38 * self.fs)  # Up to ~155 BPM
        f_vpeaks, _ = signal.find_peaks(f_vpg, distance=min_dist, prominence=np.std(f_vpg) * 0.4)
        p_vpeaks, _ = signal.find_peaks(p_vpg, distance=min_dist, prominence=np.std(p_vpg) * 0.4)

        face_beats = [b for idx in f_vpeaks if (b := self.solve_tangent_foot(f_ppg, f_vpg, idx)) is not None]
        palm_beats = [b for idx in p_vpeaks if (b := self.solve_tangent_foot(p_ppg, p_vpg, idx)) is not None]

        certified_pairs = []
        rejected_count = 0
        t_grid_full = np.arange(len(p_ppg)) / self.fs
        palm_interp = interp.interp1d(t_grid_full, p_ppg, kind="cubic", bounds_error=False, fill_value=0.0)

        for k in range(len(face_beats) - 1):
            fb = face_beats[k]
            fb_next = face_beats[k + 1]
            t_cycle = fb_next["t_foot"] - fb["t_foot"]
            if not (0.38 <= t_cycle <= 1.50):
                continue

            # Strict Gated Temporal Search Window: tau in [30 ms, 220 ms]
            candidates = [
                pb
                for pb in palm_beats
                if self.min_ptt_sec <= (pb["t_foot"] - fb["t_foot"]) <= self.max_ptt_sec
            ]
            if not candidates:
                rejected_count += 1
                continue

            pb = candidates[0]
            ptt_foot_sec = pb["t_foot"] - fb["t_foot"]
            ptt_vmax_sec = pb["t_vmax"] - fb["t_vmax"]

            # Extract single-beat waveforms and shift Palm backward by ptt_foot_sec
            num_pts = int(round(t_cycle * self.fs))
            t_beat = np.linspace(fb["t_foot"], fb["t_foot"] + t_cycle, num_pts, endpoint=False)
            face_interp = interp.interp1d(t_grid_full, f_ppg, kind="cubic", bounds_error=False, fill_value=0.0)

            w_face = face_interp(t_beat)
            w_palm_shifted = palm_interp(t_beat + ptt_foot_sec)

            # Min-max normalize both beats to [0, 1] for morphological comparison
            P_f = (w_face - np.min(w_face)) / (np.ptp(w_face) + 1e-8)
            P_p = (w_palm_shifted - np.min(w_palm_shifted)) / (np.ptp(w_palm_shifted) + 1e-8)

            povi = compute_povi(P_f, P_p)
            mda = compute_mda(P_f, P_p)
            up_len = max(5, int((fb["t_sys"] - fb["t_foot"]) * self.fs))
            r_upstroke = compute_upstroke_correlation(P_f, P_p, up_len)

            # Strict Tripartite Certification Gate
            if povi >= 0.90 and mda <= 0.12 and r_upstroke >= 0.95:
                p_start = pb["foot_idx"]
                p_end = min(len(p_apg), pb["foot_idx"] + num_pts)
                apg_waves = self.extract_apg_waves(
                    p_apg[p_start:p_end], max(1, int((pb["t_vmax"] - pb["t_foot"]) * self.fs))
                )
                certified_pairs.append({
                    "ptt_foot_ms": ptt_foot_sec * 1000.0,
                    "ptt_vmax_ms": ptt_vmax_sec * 1000.0,
                    "t_cycle_sec": t_cycle,
                    "povi": povi,
                    "mda": mda,
                    "r_upstroke": r_upstroke,
                    "face_beat_norm": P_f,
                    "palm_beat_shifted_norm": P_p,
                    "face_beat_meta": fb,
                    "palm_beat_meta": pb,
                    "apg_waves": apg_waves,
                })
            else:
                rejected_count += 1

        return {
            "certified_pairs": certified_pairs,
            "num_certified": len(certified_pairs),
            "num_rejected": rejected_count,
        }
