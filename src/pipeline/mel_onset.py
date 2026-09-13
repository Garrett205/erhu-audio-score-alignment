"""Offline-only log-Mel / SuperFlux-style evidence extractor.

This module has no import path into the production or streaming pipeline.  It
uses a fixed, recording-adaptive peak rule and writes evidence for the
isolated ablation only; it never invokes or replaces dSWIPE/CREPE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import find_peaks
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class MelOnsetConfig:
    sample_rate: int = 16000
    win_length: int = 400               # 25 ms
    hop_length: int = 80                # 5 ms
    n_fft: int = 512
    n_mels: int = 80
    fmin: float = 50.0
    fmax: float = 8000.0
    frequency_max_radius_bins: int = 3  # +/- 3 Mel bins
    temporal_lag_frames: int = 3        # 15 ms
    local_norm_window_frames: int = 201 # 1.005 s, odd
    adaptive_percentile: float = 90.0
    minimum_z: float = 2.5
    prominence_z: float = 0.5
    min_peak_distance_frames: int = 6   # 30 ms


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(config: MelOnsetConfig) -> np.ndarray:
    frequencies = np.fft.rfftfreq(config.n_fft, 1.0 / config.sample_rate)
    edges = np.linspace(_hz_to_mel(np.array([config.fmin]))[0], _hz_to_mel(np.array([config.fmax]))[0], config.n_mels + 2)
    edges_hz = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
    weights = np.zeros((config.n_mels, frequencies.size), dtype=float)
    for band in range(config.n_mels):
        left, center, right = edges_hz[band:band + 3]
        rise = (frequencies - left) / max(center - left, 1e-12)
        fall = (right - frequencies) / max(right - center, 1e-12)
        weights[band] = np.maximum(0.0, np.minimum(rise, fall))
    return weights


def _load_mono_16k(audio_path: Path, target_sr: int) -> np.ndarray:
    sr, signal = wavfile.read(audio_path)
    signal = np.asarray(signal)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float64) / max(1.0, float(np.iinfo(signal.dtype).max))
    else:
        signal = signal.astype(np.float64)
    if sr != target_sr:
        # The study inputs are already 16 kHz.  This deterministic fallback
        # retains the fixed analysis rate without adding a package dependency.
        signal = resample_poly(signal, target_sr, sr)
    return signal


def extract(audio_path: str | Path, config: MelOnsetConfig = MelOnsetConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frame-wise novelty and selected peaks on the raw 5-ms audio axis."""
    audio_path = Path(audio_path)
    y = _load_mono_16k(audio_path, config.sample_rate)
    # Match a centered STFT time grid (the reported frame time is always the
    # pipeline's raw 5-ms index, exactly as existing onset files use).
    padded = np.pad(y, (config.n_fft // 2, config.n_fft // 2), mode="reflect")
    count = 1 + (padded.size - config.n_fft) // config.hop_length
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(count, config.n_fft),
        strides=(padded.strides[0] * config.hop_length, padded.strides[0]), writeable=False,
    )
    window = np.hanning(config.win_length)
    analysis = np.zeros((count, config.n_fft), dtype=float)
    offset = (config.n_fft - config.win_length) // 2
    analysis[:, offset:offset + config.win_length] = frames[:, offset:offset + config.win_length] * window
    power = np.abs(np.fft.rfft(analysis, n=config.n_fft, axis=1)) ** 2
    mel_power = _mel_filterbank(config) @ power.T
    log_mel = np.log(np.maximum(mel_power, 1e-10))
    lagged = np.zeros_like(log_mel)
    lag = config.temporal_lag_frames
    lagged[:, lag:] = log_mel[:, :-lag]
    filtered = maximum_filter1d(
        lagged, size=2 * config.frequency_max_radius_bins + 1, axis=0,
        mode="nearest",
    )
    novelty = np.maximum(0.0, log_mel - filtered).sum(axis=0)
    local_median = median_filter(novelty, size=config.local_norm_window_frames, mode="nearest")
    local_mad = median_filter(np.abs(novelty - local_median), size=config.local_norm_window_frames, mode="nearest")
    # Bowed-string excerpts can have locally constant novelty, so a literal
    # zero MAD would make a numerical epsilon masquerade as onset evidence.
    # One recording-global robust floor keeps the rule adaptive but finite.
    deviations = np.abs(novelty - local_median)
    positive = deviations[deviations > 0]
    global_floor = float(np.percentile(positive, 25.0)) if positive.size else 1.0
    local_scale = np.maximum(1.4826 * local_mad, global_floor)
    z = np.maximum(0.0, (novelty - local_median) / local_scale)
    adaptive_height = max(config.minimum_z, float(np.percentile(z, config.adaptive_percentile)))
    peaks, properties = find_peaks(
        z, height=adaptive_height, prominence=config.prominence_z,
        distance=config.min_peak_distance_frames,
    )
    times = np.arange(z.size, dtype=float) * config.hop_length / config.sample_rate
    curves = pd.DataFrame({
        "raw_time_s": times,
        "novelty_raw": novelty,
        "novelty_norm": z,
        "local_median": local_median,
        "local_mad": local_mad,
    })
    peak_frame = pd.DataFrame({
        "raw_frame": peaks.astype(int),
        "raw_time_s": times[peaks],
        "mel_strength": z[peaks],
        "mel_novelty_raw": novelty[peaks],
        "mel_peak_height_threshold": adaptive_height,
        "mel_peak_prominence": properties.get("prominences", np.full(peaks.size, np.nan)),
    })
    return curves, peak_frame


def save_evidence(audio_path: str | Path, output_dir: str | Path, prefix: str, config: MelOnsetConfig = MelOnsetConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves, peaks = extract(audio_path, config)
    curves.to_csv(output_dir / f"{prefix}_mel_superflux_novelty.csv", index=False, encoding="utf-8-sig")
    peaks.to_csv(output_dir / f"{prefix}_mel_superflux_peaks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(config)]).to_csv(output_dir / f"{prefix}_mel_superflux_config.csv", index=False, encoding="utf-8-sig")
    return curves, peaks
