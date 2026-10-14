"""
Non-Destructive Phase-Preserving Biomedical Optical Signal Processing Suite.

Implements zero-phase biomedical optical signal processing pipeline:
  1. Multi-ROI Plane-Orthogonal-to-Skin (POS) & Chrominance-Based (CHROM) color projection.
  2. O(N) banded sparse Smoothness Priors Detrending (SPA, lambda=500.0 at 250 Hz).
  3. Zero-phase forward-backward 4th-order Butterworth bandpass [0.65 - 8.0 Hz] via sosfiltfilt.
  4. 4-level Symlet-6 (sym6) wavelet soft-thresholding (VisuShrink / MAD).
  5. Zero-phase Savitzky-Golay 1st (VPG) and 2nd (APG) derivatives.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import pywt


class NonDestructiveSignalPipeline:
    """
    Zero-phase biomedical optical signal processing pipeline:
      - POS / CHROM color projections
      - Fast O(N) banded sparse Smoothness Priors Detrending (SPA)
      - Zero-phase Butterworth [0.65 - 8.0 Hz]
      - Symlet-6 (sym6) wavelet soft-thresholding
      - Savitzky-Golay VPG (1st deriv) and APG (2nd deriv)
    """

    def __init__(self, fs: int = 250, spa_lambda: float = 500.0):
        self.fs = fs
        self.spa_lambda = spa_lambda
        # Precompute 4th-order Butterworth SOS for numerical stability
        self.sos_bp = signal.butter(
            N=4, Wn=[0.65, 8.0], btype="bandpass", fs=self.fs, output="sos"
        )

    @staticmethod
    def project_pos(rgb_trace: np.ndarray, fs: int = 250, win_sec: float = 1.6) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin (POS) projection with sliding-window overlap-add synthesis.
        rgb_trace: shape (N, 3) ordered as [R, G, B].
        """
        N = rgb_trace.shape[0]
        L = max(int(win_sec * fs), 16)
        if N < L:
            # Fallback for short sequences: simple projection
            mean_rgb = np.mean(rgb_trace, axis=0, keepdims=True) + 1e-8
            c_norm = (rgb_trace / mean_rgb) - 1.0
            p_pos = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
            s = p_pos @ c_norm.T
            alpha = np.std(s[0, :]) / (np.std(s[1, :]) + 1e-8)
            h = s[0, :] + alpha * s[1, :]
            return ((h - np.mean(h)) / (np.std(h) + 1e-8)).astype(np.float32)

        h_pos = np.zeros(N, dtype=np.float64)
        count = np.zeros(N, dtype=np.float64)
        p_pos = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

        for t in range(0, N - L + 1):
            c_win = rgb_trace[t : t + L, :]
            mean_rgb = np.mean(c_win, axis=0, keepdims=True) + 1e-8
            c_norm = (c_win / mean_rgb) - 1.0
            s = p_pos @ c_norm.T  # shape (2, L)
            std_s1 = np.std(s[0, :])
            std_s2 = np.std(s[1, :]) + 1e-8
            alpha = std_s1 / std_s2
            h_win = s[0, :] + alpha * s[1, :]
            std_h = np.std(h_win) + 1e-8
            h_win_norm = (h_win - np.mean(h_win)) / std_h
            h_pos[t : t + L] += h_win_norm
            count[t : t + L] += 1.0

        mask = count > 0
        h_pos[mask] /= count[mask]
        return h_pos.astype(np.float32)

    @staticmethod
    def project_chrom(rgb_trace: np.ndarray, fs: int = 250, win_sec: float = 1.6) -> np.ndarray:
        """
        Chrominance-based (CHROM) rPPG projection with sliding-window overlap-add synthesis.
        rgb_trace: shape (N, 3) ordered as [R, G, B].
        """
        N = rgb_trace.shape[0]
        L = max(int(win_sec * fs), 16)
        if N < L:
            mean_rgb = np.mean(rgb_trace, axis=0, keepdims=True) + 1e-8
            c_norm = (rgb_trace / mean_rgb) - 1.0
            xs = 3.0 * c_norm[:, 0] - 2.0 * c_norm[:, 1]
            ys = 1.5 * c_norm[:, 0] + 1.0 * c_norm[:, 1] - 1.5 * c_norm[:, 2]
            alpha = np.std(xs) / (np.std(ys) + 1e-8)
            h = xs - alpha * ys
            return ((h - np.mean(h)) / (np.std(h) + 1e-8)).astype(np.float32)

        h_chrom = np.zeros(N, dtype=np.float64)
        count = np.zeros(N, dtype=np.float64)

        for t in range(0, N - L + 1):
            c_win = rgb_trace[t : t + L, :]
            mean_rgb = np.mean(c_win, axis=0, keepdims=True) + 1e-8
            c_norm = (c_win / mean_rgb) - 1.0
            xs = 3.0 * c_norm[:, 0] - 2.0 * c_norm[:, 1]
            ys = 1.5 * c_norm[:, 0] + 1.0 * c_norm[:, 1] - 1.5 * c_norm[:, 2]
            alpha = np.std(xs) / (np.std(ys) + 1e-8)
            h_win = xs - alpha * ys
            std_h = np.std(h_win) + 1e-8
            h_win_norm = (h_win - np.mean(h_win)) / std_h
            h_chrom[t : t + L] += h_win_norm
            count[t : t + L] += 1.0

        mask = count > 0
        h_chrom[mask] /= count[mask]
        return h_chrom.astype(np.float32)

    def smoothness_priors_detrend(self, x: np.ndarray) -> np.ndarray:
        """
        Solves z = (I + lambda^2 * D2^T * D2)^(-1) x in O(N) time using CSC sparse solver.
        Eliminates low-frequency drift without introducing any phase delay.
        """
        N = len(x)
        if N < 6:
            return x.astype(np.float32)
        ones = np.ones(N, dtype=np.float64)
        d2 = sparse.spdiags([ones, -2.0 * ones, ones], [0, 1, 2], N - 2, N, format="csc")
        i_mat = sparse.eye(N, format="csc")
        a = i_mat + (self.spa_lambda ** 2) * (d2.T @ d2)
        trend_z = splinalg.spsolve(a, x.astype(np.float64))
        return (x - trend_z).astype(np.float32)

    def zero_phase_bandpass(self, x: np.ndarray) -> np.ndarray:
        """Applies forward-backward 4th-order Butterworth [0.65 - 8.0 Hz] via sosfiltfilt."""
        padlen = min(3 * 9, len(x) - 1)
        if padlen < 3:
            return x.astype(np.float32)
        return signal.sosfiltfilt(self.sos_bp, x.astype(np.float64), padlen=padlen).astype(np.float32)

    def symlet6_wavelet_denoise(self, x: np.ndarray) -> np.ndarray:
        """
        Applies 4-level Symlet-6 (sym6) discrete wavelet denoising.
        Soft-thresholds high-frequency noise in D1, D2, D3 while preserving cardiac D4, A4.
        """
        N = len(x)
        if N < 32:
            return x.astype(np.float32)
        coeffs = pywt.wavedec(x, wavelet="sym6", mode="symmetric", level=4)
        for idx in [-1, -2, -3]:  # D1, D2, D3
            detail = coeffs[idx]
            sigma_mad = np.median(np.abs(detail)) / 0.6745
            thresh = sigma_mad * np.sqrt(2.0 * np.log(max(N, 2)))
            if idx in [-1, -2]:
                coeffs[idx] = pywt.threshold(detail, value=thresh * 1.5, mode="soft")
            else:
                coeffs[idx] = pywt.threshold(detail, value=thresh * 0.8, mode="soft")

        reconstructed = pywt.waverec(coeffs, wavelet="sym6", mode="symmetric")
        return reconstructed[:N].astype(np.float32)

    def compute_vpg_apg(self, ppg_clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes zero-phase 1st (VPG) and 2nd (APG) derivatives via Savitzky-Golay filter."""
        win_len = int(0.06 * self.fs) | 1  # Odd window (~15 samples at 250 Hz)
        win_len = max(win_len, 7)
        if len(ppg_clean) <= win_len:
            # Fallback for very short signals
            dt = 1.0 / self.fs
            vpg = np.gradient(ppg_clean, dt)
            apg = np.gradient(vpg, dt)
            return vpg.astype(np.float32), apg.astype(np.float32)

        dt = 1.0 / self.fs
        vpg = signal.savgol_filter(ppg_clean, window_length=win_len, polyorder=4, deriv=1, delta=dt)
        apg = signal.savgol_filter(ppg_clean, window_length=win_len, polyorder=4, deriv=2, delta=dt)
        return vpg.astype(np.float32), apg.astype(np.float32)

    def process_waveform(self, raw_signal: np.ndarray) -> dict[str, np.ndarray]:
        """Runs the complete non-destructive zero-phase pipeline."""
        x_detrend = self.smoothness_priors_detrend(raw_signal)
        x_bp = self.zero_phase_bandpass(x_detrend)
        x_clean = self.symlet6_wavelet_denoise(x_bp)
        vpg, apg = self.compute_vpg_apg(x_clean)
        return {"ppg": x_clean, "vpg": vpg, "apg": apg}
