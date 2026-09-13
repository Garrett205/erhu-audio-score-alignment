from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import math
import time

import numpy as np
import pandas as pd

SCORE_PITCH_COL = "音高频率(Hz)"
SCORE_TIME_COL = "累计时间(s)"
SCORE_DUR_COLS = ("逻辑持续时间(s)", "持续时间(s)")
SCORE_ID_COL = "score_note_id"
SCORE_USE_COL = "参与起音对齐"

PITCH_TIME_COL = "时间(s)"
PITCH_F0_COL = "onset频率(Hz)"
ONSET_TIME_COL = "时间(s)"
ONSET_RAW_COL = "onset_strength"
ONSET_NORM_COL = "onset_strength_norm"


@dataclass
class PitchOnsetDTWConfig:
    """Frozen baseline parameters.

    The algorithm is intentionally generic: framewise score pitch + score-onset
    envelope are aligned to framewise performance F0 + raw acoustic onset
    strength.  No project onset candidates, OS, pitch2, or DP outputs are used.
    """

    hop_ms: float = 20.0
    pitch_weight: float = 1.0
    onset_weight: float = 0.5
    pitch_clip_cents: float = 200.0
    onset_sigma_ms: float = 30.0
    performance_onset_smooth_ms: float = 20.0
    onset_norm_percentile: float = 95.0
    band_frac: float = 0.35
    anchor_rule: str = "best_local"  # best_local | first | median
    unvoiced_mismatch_cost: float = 1.0
    both_unvoiced_cost: float = 0.0
    performance_edge_pad_ms: float = 0.0

    def validate(self) -> None:
        if self.hop_ms <= 0:
            raise ValueError("hop_ms must be > 0")
        if self.pitch_weight < 0 or self.onset_weight < 0:
            raise ValueError("feature weights must be >= 0")
        if self.pitch_clip_cents <= 0:
            raise ValueError("pitch_clip_cents must be > 0")
        if self.onset_sigma_ms <= 0:
            raise ValueError("onset_sigma_ms must be > 0")
        if self.performance_onset_smooth_ms < 0:
            raise ValueError("performance_onset_smooth_ms must be >= 0")
        if not (0 < self.onset_norm_percentile <= 100):
            raise ValueError("onset_norm_percentile must be in (0, 100]")
        if not (0 < self.band_frac <= 1.0):
            raise ValueError("band_frac must be in (0, 1]")
        if self.anchor_rule not in {"best_local", "first", "median"}:
            raise ValueError(f"unsupported anchor_rule={self.anchor_rule!r}")


@dataclass
class RunSummary:
    piece_id: str
    n_score_events: int
    n_score_frames: int
    n_performance_frames: int
    n_mapped_events: int
    n_unmapped_events: int
    n_duplicate_anchor_times: int
    path_length: int
    total_path_cost: float
    normalized_path_cost: float
    band_frac: float
    hop_ms: float
    elapsed_s: float

    def to_dict(self):
        return asdict(self)


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def hz_to_midi(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    out[valid] = 69.0 + 12.0 * np.log2(x[valid] / 440.0)
    return out


def cents_error(pred_hz: np.ndarray, ref_hz: np.ndarray) -> np.ndarray:
    p = np.asarray(pred_hz, dtype=float)
    r = np.asarray(ref_hz, dtype=float)
    out = np.full(np.broadcast_shapes(p.shape, r.shape), np.nan, dtype=float)
    pb = np.broadcast_to(p, out.shape)
    rb = np.broadcast_to(r, out.shape)
    valid = np.isfinite(pb) & np.isfinite(rb) & (pb > 0) & (rb > 0)
    out[valid] = 1200.0 * np.log2(pb[valid] / rb[valid])
    return out


def load_score(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path).copy()
    if SCORE_USE_COL in df.columns:
        mask = df[SCORE_USE_COL].astype(str).str.strip().eq("是")
        df = df.loc[mask].copy()

    needed = {SCORE_ID_COL, SCORE_PITCH_COL, SCORE_TIME_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"score CSV missing columns: {sorted(missing)}")

    df[SCORE_ID_COL] = df[SCORE_ID_COL].astype(str)
    df["score_pitch_hz"] = pd.to_numeric(df[SCORE_PITCH_COL], errors="coerce")
    df["score_time_s"] = pd.to_numeric(df[SCORE_TIME_COL], errors="coerce")
    dur_col = next((c for c in SCORE_DUR_COLS if c in df.columns), None)
    df["score_duration_s"] = pd.to_numeric(df[dur_col], errors="coerce") if dur_col else np.nan

    df = df[
        df["score_pitch_hz"].gt(0)
        & df["score_pitch_hz"].notna()
        & df["score_time_s"].notna()
    ].copy()
    df = df.sort_values(["score_time_s"], kind="stable").reset_index(drop=True)
    df.insert(0, "score_index", np.arange(len(df), dtype=int))
    if df.empty:
        raise ValueError("score contains no alignable positive-pitch events")
    if df[SCORE_ID_COL].duplicated().any():
        dup = df.loc[df[SCORE_ID_COL].duplicated(keep=False), SCORE_ID_COL].tolist()[:10]
        raise ValueError(f"duplicate score_note_id values after filtering: {dup}")
    return df


def load_performance_pitch(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path).copy()
    needed = {PITCH_TIME_COL, PITCH_F0_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"pitch CSV missing columns: {sorted(missing)}")
    df["time_s"] = pd.to_numeric(df[PITCH_TIME_COL], errors="coerce")
    df["pitch_hz"] = pd.to_numeric(df[PITCH_F0_COL], errors="coerce")
    df = df[df["time_s"].notna()].copy()
    df["pitch_hz"] = df["pitch_hz"].where(df["pitch_hz"].gt(0), 0.0).fillna(0.0)
    df = df.sort_values("time_s", kind="stable").drop_duplicates("time_s", keep="first").reset_index(drop=True)
    if df.empty or not df["pitch_hz"].gt(0).any():
        raise ValueError("performance pitch CSV contains no positive F0 frames")
    return df


def robust_normalize_onset(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    if not valid.any():
        return out
    scale = float(np.percentile(x[valid], percentile))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(x[valid]))
    if not np.isfinite(scale) or scale <= 1e-12:
        return out
    out[valid] = np.clip(x[valid] / scale, 0.0, 1.0)
    return out


def load_onset_strength(path: str | Path, cfg: PitchOnsetDTWConfig) -> pd.DataFrame:
    df = _read_csv(path).copy()
    if ONSET_TIME_COL not in df.columns:
        raise ValueError(f"onset CSV missing column: {ONSET_TIME_COL}")
    df["time_s"] = pd.to_numeric(df[ONSET_TIME_COL], errors="coerce")

    # Prefer the raw spectral-flux-like envelope and normalize here so every
    # piece follows one recorded rule.  Fall back to the pre-normalized column.
    if ONSET_RAW_COL in df.columns:
        raw = pd.to_numeric(df[ONSET_RAW_COL], errors="coerce").to_numpy(float)
        norm = robust_normalize_onset(raw, cfg.onset_norm_percentile)
        source_col = ONSET_RAW_COL
    elif ONSET_NORM_COL in df.columns:
        raw = pd.to_numeric(df[ONSET_NORM_COL], errors="coerce").to_numpy(float)
        norm = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 1.0)
        source_col = ONSET_NORM_COL
    else:
        raise ValueError(f"onset CSV needs {ONSET_RAW_COL!r} or {ONSET_NORM_COL!r}")

    out = pd.DataFrame({"time_s": df["time_s"], "onset_strength": norm})
    out = out[out["time_s"].notna()].sort_values("time_s", kind="stable").drop_duplicates("time_s", keep="first").reset_index(drop=True)
    out.attrs["source_column"] = source_col
    return out


def compute_onset_strength_from_audio(audio_path: str | Path, cfg: PitchOnsetDTWConfig) -> pd.DataFrame:
    """Generic raw acoustic onset strength fallback.

    This deliberately reproduces only the raw PMSDB-style onset envelope.  It
    does NOT call the project's onset.py and does NOT create energy/yellow/
    plateau candidates.
    """
    try:
        import librosa
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "librosa is required only when no *_onset强度_5ms.csv exists; "
            "reuse the current dachuang environment rather than rebuilding it"
        ) from exc

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    hop = 80  # exactly 5 ms at 16 kHz, matching current PMSDB output
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = np.arange(len(env), dtype=float) * (hop / float(sr))
    norm = robust_normalize_onset(np.asarray(env, dtype=float), cfg.onset_norm_percentile)
    out = pd.DataFrame({"time_s": times, "onset_strength": norm})
    out.attrs["source_column"] = "librosa.onset.onset_strength"
    return out


def _gaussian_kernel(sigma_frames: float) -> np.ndarray:
    if sigma_frames <= 0.25:
        return np.array([1.0], dtype=float)
    radius = max(1, int(math.ceil(4.0 * sigma_frames)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_frames) ** 2)
    k /= np.sum(k)
    return k


def _smooth_1d(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sigma_frames <= 0.25 or x.size < 3:
        return x.copy()
    k = _gaussian_kernel(sigma_frames)
    pad = len(k) // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _score_onset_envelope(grid_t: np.ndarray, onset_times: np.ndarray, sigma_s: float) -> np.ndarray:
    env = np.zeros(len(grid_t), dtype=float)
    if len(grid_t) == 0:
        return env
    hop = float(np.median(np.diff(grid_t))) if len(grid_t) > 1 else sigma_s
    radius_frames = max(1, int(math.ceil(4.0 * sigma_s / max(hop, 1e-9))))
    for onset in onset_times:
        center = int(np.argmin(np.abs(grid_t - float(onset))))
        lo = max(0, center - radius_frames)
        hi = min(len(grid_t), center + radius_frames + 1)
        dt = grid_t[lo:hi] - float(onset)
        pulse = np.exp(-0.5 * (dt / sigma_s) ** 2)
        env[lo:hi] = np.maximum(env[lo:hi], pulse)
    return np.clip(env, 0.0, 1.0)


def _infer_event_durations(score: pd.DataFrame, hop_s: float) -> np.ndarray:
    onset = score["score_time_s"].to_numpy(float)
    dur = score["score_duration_s"].to_numpy(float)
    out = dur.copy()
    for i in range(len(out)):
        if not np.isfinite(out[i]) or out[i] <= 0:
            if i + 1 < len(out) and onset[i + 1] > onset[i]:
                out[i] = onset[i + 1] - onset[i]
            else:
                out[i] = hop_s
    return np.maximum(out, hop_s)


def build_score_features(score: pd.DataFrame, cfg: PitchOnsetDTWConfig):
    hop_s = cfg.hop_ms / 1000.0
    onset = score["score_time_s"].to_numpy(float)
    dur = _infer_event_durations(score, hop_s)
    pitch = score["score_pitch_hz"].to_numpy(float)
    start = float(np.min(onset))
    end = float(np.max(onset + dur))
    if end <= start:
        end = start + hop_s
    grid = np.arange(start, end + 0.5 * hop_s, hop_s, dtype=float)

    pitch_hz = np.full(len(grid), np.nan, dtype=float)
    # Monophonic score template.  Later score events overwrite overlaps; this is
    # deterministic and intentionally leaves simultaneous/grace ambiguity to DTW.
    for t0, d, f0 in zip(onset, dur, pitch):
        t1 = float(t0 + max(d, hop_s))
        lo = int(np.searchsorted(grid, t0 - 0.5 * hop_s, side="left"))
        hi = int(np.searchsorted(grid, t1 - 0.5 * hop_s, side="left"))
        lo = max(0, min(lo, len(grid) - 1))
        hi = max(lo + 1, min(hi, len(grid)))
        pitch_hz[lo:hi] = float(f0)

    onset_env = _score_onset_envelope(grid, onset, cfg.onset_sigma_ms / 1000.0)
    event_grid_index = np.array([int(np.argmin(np.abs(grid - t))) for t in onset], dtype=int)
    return {
        "time_s": grid,
        "pitch_hz": pitch_hz,
        "pitch_midi": hz_to_midi(pitch_hz),
        "onset": onset_env,
        "event_grid_index": event_grid_index,
        "event_duration_s": dur,
    }


def _nearest_resample(src_t: np.ndarray, src_x: np.ndarray, dst_t: np.ndarray, max_gap_s: float) -> np.ndarray:
    src_t = np.asarray(src_t, dtype=float)
    src_x = np.asarray(src_x, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)
    out = np.zeros(len(dst_t), dtype=float)
    if len(src_t) == 0:
        return out
    pos = np.searchsorted(src_t, dst_t, side="left")
    for k, p in enumerate(pos):
        cand = []
        if p < len(src_t):
            cand.append(p)
        if p > 0:
            cand.append(p - 1)
        if not cand:
            continue
        j = min(cand, key=lambda jj: abs(src_t[jj] - dst_t[k]))
        if abs(src_t[j] - dst_t[k]) <= max_gap_s:
            out[k] = src_x[j]
    return out


def build_performance_features(
    pitch_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    cfg: PitchOnsetDTWConfig,
):
    hop_s = cfg.hop_ms / 1000.0
    pt = pitch_df["time_s"].to_numpy(float)
    ph = pitch_df["pitch_hz"].to_numpy(float)
    valid = np.isfinite(ph) & (ph > 0)
    if not valid.any():
        raise ValueError("performance has no valid positive pitch")

    edge_pad = cfg.performance_edge_pad_ms / 1000.0
    start = max(float(np.min(pt)), float(pt[np.flatnonzero(valid)[0]]) - edge_pad)
    end = min(float(np.max(pt)), float(pt[np.flatnonzero(valid)[-1]]) + edge_pad)
    if end <= start:
        end = start + hop_s
    grid = np.arange(start, end + 0.5 * hop_s, hop_s, dtype=float)

    dt = np.diff(pt)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    src_hop = float(np.median(finite_dt)) if len(finite_dt) else hop_s
    pitch_hz = _nearest_resample(pt, ph, grid, max_gap_s=max(2.5 * src_hop, 1.5 * hop_s))
    pitch_hz[~np.isfinite(pitch_hz)] = 0.0

    ot = onset_df["time_s"].to_numpy(float)
    ox = onset_df["onset_strength"].to_numpy(float)
    if len(ot) >= 2:
        onset = np.interp(grid, ot, ox, left=0.0, right=0.0)
    else:
        onset = np.zeros(len(grid), dtype=float)
    onset = np.clip(np.nan_to_num(onset, nan=0.0), 0.0, 1.0)
    smooth_sigma_frames = (cfg.performance_onset_smooth_ms / 1000.0) / hop_s
    onset = _smooth_1d(onset, smooth_sigma_frames)
    if np.max(onset) > 0:
        onset = np.clip(onset / np.max(onset), 0.0, 1.0)

    return {
        "time_s": grid,
        "pitch_hz": pitch_hz,
        "pitch_midi": hz_to_midi(pitch_hz),
        "onset": onset,
    }


def local_cost_components(
    score_pitch_midi: float,
    score_onset: float,
    perf_pitch_midi: float,
    perf_onset: float,
    cfg: PitchOnsetDTWConfig,
):
    sv = np.isfinite(score_pitch_midi)
    pv = np.isfinite(perf_pitch_midi)
    if sv and pv:
        cents = abs(float(score_pitch_midi - perf_pitch_midi)) * 100.0
        pitch_cost = min(cents / cfg.pitch_clip_cents, 1.0) ** 2
    elif (not sv) and (not pv):
        pitch_cost = float(cfg.both_unvoiced_cost)
    else:
        pitch_cost = float(cfg.unvoiced_mismatch_cost)
    onset_cost = (float(score_onset) - float(perf_onset)) ** 2
    total = cfg.pitch_weight * pitch_cost + cfg.onset_weight * onset_cost
    return float(pitch_cost), float(onset_cost), float(total)


# Numba is already a dependency of librosa in the user's current environment.
# cache=False is intentional: this package must not create/reuse project-level
# Numba caches and must not disturb the validated STARS cache workaround.
try:  # pragma: no cover - whether numba is present is environment-specific
    from numba import njit

    @njit(cache=False)
    def _banded_dtw_numba(
        sp, so, pp, po,
        pitch_weight, onset_weight, pitch_clip_cents,
        unvoiced_mismatch_cost, both_unvoiced_cost, band_frac,
    ):
        n = sp.shape[0]
        m = pp.shape[0]
        inf = np.inf
        prev = np.full(m, inf, dtype=np.float64)
        cur = np.full(m, inf, dtype=np.float64)
        back = np.full((n, m), 255, dtype=np.uint8)
        radius = max(2, int(math.ceil(band_frac * max(n, m))))

        for i in range(n):
            for jj in range(m):
                cur[jj] = inf
            if n <= 1:
                center = 0
            else:
                center = int(round(i * (m - 1) / (n - 1)))
            jlo = max(0, center - radius)
            jhi = min(m - 1, center + radius)
            if i == 0:
                jlo = 0
            if i == n - 1:
                jhi = m - 1

            for j in range(jlo, jhi + 1):
                sv = not np.isnan(sp[i])
                pv = not np.isnan(pp[j])
                if sv and pv:
                    cents = abs(sp[i] - pp[j]) * 100.0
                    ratio = cents / pitch_clip_cents
                    if ratio > 1.0:
                        ratio = 1.0
                    pcost = ratio * ratio
                elif (not sv) and (not pv):
                    pcost = both_unvoiced_cost
                else:
                    pcost = unvoiced_mismatch_cost
                od = so[i] - po[j]
                ocost = od * od
                local = pitch_weight * pcost + onset_weight * ocost

                if i == 0 and j == 0:
                    cur[j] = local
                    back[i, j] = 3
                    continue

                diag = inf
                up = inf
                left = inf
                if i > 0 and j > 0:
                    diag = prev[j - 1]
                if i > 0:
                    up = prev[j]
                if j > 0:
                    left = cur[j - 1]

                if diag <= up and diag <= left:
                    best = diag
                    code = 0
                elif up <= left:
                    best = up
                    code = 1
                else:
                    best = left
                    code = 2
                if not np.isinf(best):
                    cur[j] = local + best
                    back[i, j] = code
            tmp = prev
            prev = cur
            cur = tmp

        total = prev[m - 1]
        return back, total

except Exception:  # pragma: no cover
    _banded_dtw_numba = None


def _banded_dtw_python(
    sp: np.ndarray, so: np.ndarray, pp: np.ndarray, po: np.ndarray, cfg: PitchOnsetDTWConfig
):
    n, m = len(sp), len(pp)
    inf = float("inf")
    prev = np.full(m, inf, dtype=float)
    cur = np.full(m, inf, dtype=float)
    back = np.full((n, m), 255, dtype=np.uint8)
    radius = max(2, int(math.ceil(cfg.band_frac * max(n, m))))

    for i in range(n):
        cur.fill(inf)
        center = 0 if n <= 1 else int(round(i * (m - 1) / (n - 1)))
        jlo = max(0, center - radius)
        jhi = min(m - 1, center + radius)
        if i == 0:
            jlo = 0
        if i == n - 1:
            jhi = m - 1
        for j in range(jlo, jhi + 1):
            _, _, local = local_cost_components(sp[i], so[i], pp[j], po[j], cfg)
            if i == 0 and j == 0:
                cur[j] = local
                back[i, j] = 3
                continue
            diag = prev[j - 1] if i > 0 and j > 0 else inf
            up = prev[j] if i > 0 else inf
            left = cur[j - 1] if j > 0 else inf
            if diag <= up and diag <= left:
                best, code = diag, 0
            elif up <= left:
                best, code = up, 1
            else:
                best, code = left, 2
            if math.isfinite(best):
                cur[j] = local + best
                back[i, j] = code
        prev, cur = cur, prev
    return back, float(prev[m - 1])


def compute_banded_dtw_path(
    score_pitch_midi: np.ndarray,
    score_onset: np.ndarray,
    perf_pitch_midi: np.ndarray,
    perf_onset: np.ndarray,
    cfg: PitchOnsetDTWConfig,
):
    cfg.validate()
    sp = np.asarray(score_pitch_midi, dtype=np.float64)
    so = np.asarray(score_onset, dtype=np.float64)
    pp = np.asarray(perf_pitch_midi, dtype=np.float64)
    po = np.asarray(perf_onset, dtype=np.float64)
    n, m = len(sp), len(pp)
    if n == 0 or m == 0:
        raise ValueError("empty DTW feature sequence")
    if len(so) != n or len(po) != m:
        raise ValueError("pitch/onset feature length mismatch")

    if _banded_dtw_numba is not None:
        back, total = _banded_dtw_numba(
            sp, so, pp, po,
            float(cfg.pitch_weight), float(cfg.onset_weight), float(cfg.pitch_clip_cents),
            float(cfg.unvoiced_mismatch_cost), float(cfg.both_unvoiced_cost), float(cfg.band_frac),
        )
        backend = "numba_banded_exact"
    else:
        back, total = _banded_dtw_python(sp, so, pp, po, cfg)
        backend = "python_banded_exact"

    if not np.isfinite(total):
        raise RuntimeError(
            "No finite DTW path reached the endpoint. Increase --band-frac; "
            "do not add OS/DP information to rescue the baseline."
        )

    i, j = n - 1, m - 1
    rev: list[tuple[int, int]] = []
    guard = n + m + n * m
    steps = 0
    while True:
        rev.append((i, j))
        code = int(back[i, j])
        if code == 3:
            break
        if code == 0:
            i -= 1
            j -= 1
        elif code == 1:
            i -= 1
        elif code == 2:
            j -= 1
        else:
            raise RuntimeError(f"invalid/unreachable DTW backpointer at {(i, j)}: {code}")
        if i < 0 or j < 0:
            raise RuntimeError("DTW backtracking escaped matrix")
        steps += 1
        if steps > guard:
            raise RuntimeError("DTW backtracking guard exceeded")
    rev.reverse()
    return np.asarray(rev, dtype=np.int32), float(total), backend


def _group_path_by_score_frame(path: np.ndarray, n_score_frames: int) -> list[list[int]]:
    grouped: list[list[int]] = [[] for _ in range(n_score_frames)]
    for si, pj in np.asarray(path, dtype=int):
        if 0 <= si < n_score_frames:
            grouped[int(si)].append(int(pj))
    return grouped


def run_piece(
    score_csv: str | Path,
    pitch_csv: str | Path,
    piece_id: str,
    onset_csv: str | Path | None = None,
    audio_path: str | Path | None = None,
    cfg: PitchOnsetDTWConfig | None = None,
):
    cfg = cfg or PitchOnsetDTWConfig()
    cfg.validate()
    t0 = time.perf_counter()

    score = load_score(score_csv)
    pitch_df = load_performance_pitch(pitch_csv)
    if onset_csv is not None:
        onset_df = load_onset_strength(onset_csv, cfg)
        onset_source = f"csv:{Path(onset_csv).name}"
    else:
        if audio_path is None:
            raise ValueError("need onset_csv or audio_path")
        onset_df = compute_onset_strength_from_audio(audio_path, cfg)
        onset_source = f"audio:{Path(audio_path).name}:librosa_raw_onset_strength"

    sf = build_score_features(score, cfg)
    pf = build_performance_features(pitch_df, onset_df, cfg)
    path, total_cost, backend = compute_banded_dtw_path(
        sf["pitch_midi"], sf["onset"], pf["pitch_midi"], pf["onset"], cfg
    )

    grouped = _group_path_by_score_frame(path, len(sf["time_s"]))
    perf_t = pf["time_s"]
    perf_hz = pf["pitch_hz"]
    perf_midi = pf["pitch_midi"]
    perf_on = pf["onset"]

    rows = []
    previous_anchor_j = -1
    for i, row in score.iterrows():
        sgi = int(sf["event_grid_index"][i])
        inds = grouped[sgi]
        if inds:
            arr = np.asarray(sorted(set(inds)), dtype=int)
            first_j = int(arr[0])
            last_j = int(arr[-1])
            if cfg.anchor_rule == "first":
                anchor_j = first_j
            elif cfg.anchor_rule == "median":
                anchor_j = int(arr[len(arr) // 2])
            else:
                # Multiple score events can quantize to the same score grid
                # frame.  Choose best_local only from path cells that preserve
                # score-event order; otherwise independent readout can make a
                # later event step backward on an otherwise monotone DTW path.
                eligible = arr[arr >= previous_anchor_j] if previous_anchor_j >= 0 else arr
                if len(eligible) == 0:
                    raise RuntimeError(
                        "no monotone best_local anchor remains for score event "
                        f"{row[SCORE_ID_COL]}"
                    )
                event_midi = float(hz_to_midi([float(row["score_pitch_hz"])])[0])
                best = None
                for j in eligible:
                    pc, oc, tc = local_cost_components(event_midi, 1.0, perf_midi[j], perf_on[j], cfg)
                    key = (tc, -float(perf_on[j]), int(j))
                    if best is None or key < best[0]:
                        best = (key, int(j), pc, oc, tc)
                assert best is not None
                anchor_j = best[1]
            anchor_time = float(perf_t[anchor_j])
            anchor_pitch = float(perf_hz[anchor_j]) if perf_hz[anchor_j] > 0 else np.nan
            anchor_onset = float(perf_on[anchor_j])
            pc, oc, tc = local_cost_components(
                float(hz_to_midi([float(row["score_pitch_hz"])])[0]),
                1.0,
                perf_midi[anchor_j],
                perf_on[anchor_j],
                cfg,
            )
            first_time = float(perf_t[first_j])
            last_time = float(perf_t[last_j])
            path_count = int(len(arr))
            previous_anchor_j = anchor_j
        else:
            first_j = last_j = anchor_j = -1
            anchor_time = anchor_pitch = anchor_onset = np.nan
            pc = oc = tc = np.nan
            first_time = last_time = np.nan
            path_count = 0

        ce = cents_error(np.array([anchor_pitch]), np.array([float(row["score_pitch_hz"])]))[0]
        rows.append({
            "piece_id": piece_id,
            "method": "Pitch+Onset DTW",
            "score_index": int(row["score_index"]),
            "score_note_id": str(row[SCORE_ID_COL]),
            "score_time_s": float(row["score_time_s"]),
            "score_duration_s": float(sf["event_duration_s"][i]),
            "score_pitch_hz": float(row["score_pitch_hz"]),
            "score_grid_index": sgi,
            "score_grid_time_s": float(sf["time_s"][sgi]),
            "dtw_anchor_time_s": anchor_time,
            "dtw_anchor_pitch_hz": anchor_pitch,
            "dtw_anchor_onset_strength": anchor_onset,
            "dtw_anchor_pitch_error_cents": float(ce) if np.isfinite(ce) else np.nan,
            "dtw_anchor_pitch_cost": pc,
            "dtw_anchor_onset_cost": oc,
            "dtw_anchor_total_local_cost": tc,
            "dtw_first_time_s": first_time,
            "dtw_last_time_s": last_time,
            "dtw_span_ms": (last_time - first_time) * 1000.0 if np.isfinite(first_time) and np.isfinite(last_time) else np.nan,
            "dtw_path_points_for_score_onset_frame": path_count,
            "dtw_first_perf_index": first_j if first_j >= 0 else np.nan,
            "dtw_anchor_perf_index": anchor_j if anchor_j >= 0 else np.nan,
            "dtw_last_perf_index": last_j if last_j >= 0 else np.nan,
            "status": "dtw_aligned" if inds else "dtw_unmapped",
            "manual_correct": "",
            "manual_error_type": "",
            "manual_note": "",
        })

    out = pd.DataFrame(rows)
    mapped = out["dtw_anchor_time_s"].notna()
    duplicate_times = int(out.loc[mapped, "dtw_anchor_time_s"].duplicated(keep=False).sum())
    elapsed = time.perf_counter() - t0
    summary = RunSummary(
        piece_id=piece_id,
        n_score_events=len(score),
        n_score_frames=len(sf["time_s"]),
        n_performance_frames=len(pf["time_s"]),
        n_mapped_events=int(mapped.sum()),
        n_unmapped_events=int((~mapped).sum()),
        n_duplicate_anchor_times=duplicate_times,
        path_length=len(path),
        total_path_cost=float(total_cost),
        normalized_path_cost=float(total_cost / max(1, len(path))),
        band_frac=float(cfg.band_frac),
        hop_ms=float(cfg.hop_ms),
        elapsed_s=float(elapsed),
    )

    meta = {
        "backend": backend,
        "onset_source": onset_source,
        "config": asdict(cfg),
    }
    return out, path, pf, sf, score, summary, meta
