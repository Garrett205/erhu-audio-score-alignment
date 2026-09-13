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


@dataclass
class DTWConfig:
    backend: str = "auto"  # auto | dtaidistance | numpy
    anchor_rule: str = "first"  # first | median
    remove_unvoiced: bool = True
    pitch_unit: str = "midi"
    save_path: bool = True


@dataclass
class RunSummary:
    piece_id: str
    n_score_events: int
    n_performance_frames: int
    n_mapped_events: int
    n_unmapped_events: int
    n_duplicate_anchor_times: int
    path_length: int
    dtw_distance: float
    normalized_path_cost: float
    backend: str
    elapsed_s: float

    def to_dict(self):
        return asdict(self)


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


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def load_score(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path).copy()
    if SCORE_USE_COL in df.columns:
        mask = df[SCORE_USE_COL].astype(str).str.strip().eq("是")
        df = df.loc[mask].copy()
    needed = {SCORE_ID_COL, SCORE_PITCH_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"score CSV missing columns: {sorted(missing)}")

    df[SCORE_ID_COL] = df[SCORE_ID_COL].astype(str)
    df["score_pitch_hz"] = pd.to_numeric(df[SCORE_PITCH_COL], errors="coerce")
    if SCORE_TIME_COL in df.columns:
        df["score_time_s"] = pd.to_numeric(df[SCORE_TIME_COL], errors="coerce")
    else:
        df["score_time_s"] = np.nan

    dur_col = next((c for c in SCORE_DUR_COLS if c in df.columns), None)
    df["score_duration_s"] = pd.to_numeric(df[dur_col], errors="coerce") if dur_col else np.nan
    df = df[df["score_pitch_hz"].gt(0) & df["score_pitch_hz"].notna()].reset_index(drop=True)
    df.insert(0, "score_index", np.arange(len(df), dtype=int))
    return df


def load_performance_pitch(path: str | Path, remove_unvoiced: bool = True) -> pd.DataFrame:
    df = _read_csv(path).copy()
    needed = {PITCH_TIME_COL, PITCH_F0_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"pitch CSV missing columns: {sorted(missing)}")
    df["time_s"] = pd.to_numeric(df[PITCH_TIME_COL], errors="coerce")
    df["pitch_hz"] = pd.to_numeric(df[PITCH_F0_COL], errors="coerce")
    df = df[df["time_s"].notna()].copy()
    if remove_unvoiced:
        df = df[df["pitch_hz"].gt(0) & df["pitch_hz"].notna()].copy()
    else:
        # Classic scalar DTW cannot consume NaN/zero pitch; keep only valid frames.
        df = df[df["pitch_hz"].gt(0) & df["pitch_hz"].notna()].copy()
    df = df.sort_values("time_s").reset_index(drop=True)
    df.insert(0, "perf_index", np.arange(len(df), dtype=int))
    return df


def _dtw_dtaidistance(a: np.ndarray, b: np.ndarray):
    from dtaidistance import dtw
    path = dtw.warping_path(a.astype(np.double), b.astype(np.double))
    # dtaidistance distance uses the same squared local-cost family; keep one scalar summary.
    distance = float(dtw.distance(a.astype(np.double), b.astype(np.double)))
    return [(int(i), int(j)) for i, j in path], distance, "dtaidistance"


def _dtw_numpy_exact(a: np.ndarray, b: np.ndarray):
    """Exact unconstrained scalar DTW for short score-event sequence vs long F0 sequence.

    This is intentionally simple and dependency-free. Memory is O(len(a)*len(b));
    for the five-paper-song protocol len(a) <= 176, so it is practical.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("empty DTW sequence")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("DTW inputs must be finite")

    inf = np.float64(np.inf)
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    prev = np.full((n + 1, m + 1), 255, dtype=np.uint8)
    dp[0, 0] = 0.0

    # transitions: 0 diag, 1 up(score advances), 2 left(performance advances)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = ai - b[j - 1]
            cost = d * d
            diag = dp[i - 1, j - 1]
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            if diag <= up and diag <= left:
                dp[i, j] = cost + diag
                prev[i, j] = 0
            elif up <= left:
                dp[i, j] = cost + up
                prev[i, j] = 1
            else:
                dp[i, j] = cost + left
                prev[i, j] = 2

    i, j = n, m
    rev = []
    while i > 0 and j > 0:
        rev.append((i - 1, j - 1))
        p = int(prev[i, j])
        if p == 0:
            i -= 1; j -= 1
        elif p == 1:
            i -= 1
        elif p == 2:
            j -= 1
        else:
            raise RuntimeError(f"invalid DTW backpointer at {(i, j)}: {p}")
    while i > 0:
        rev.append((i - 1, 0)); i -= 1
    while j > 0:
        rev.append((0, j - 1)); j -= 1
    rev.reverse()
    return rev, float(math.sqrt(dp[n, m])), "numpy_exact"


def compute_dtw_path(a: np.ndarray, b: np.ndarray, backend: str = "auto"):
    if backend not in {"auto", "dtaidistance", "numpy"}:
        raise ValueError(f"unsupported backend={backend!r}")
    if backend in {"auto", "dtaidistance"}:
        try:
            return _dtw_dtaidistance(a, b)
        except Exception:
            if backend == "dtaidistance":
                raise
    return _dtw_numpy_exact(a, b)


def _event_ranges_from_path(path: list[tuple[int, int]], n_events: int) -> list[list[int]]:
    by_event: list[list[int]] = [[] for _ in range(n_events)]
    for si, pj in path:
        if 0 <= si < n_events:
            by_event[si].append(int(pj))
    return by_event


def run_piece(score_csv: str | Path, pitch_csv: str | Path, piece_id: str, cfg: DTWConfig | None = None):
    cfg = cfg or DTWConfig()
    t0 = time.perf_counter()
    score = load_score(score_csv)
    perf = load_performance_pitch(pitch_csv, cfg.remove_unvoiced)

    score_seq = hz_to_midi(score["score_pitch_hz"].to_numpy(float))
    perf_seq = hz_to_midi(perf["pitch_hz"].to_numpy(float))
    if not np.isfinite(score_seq).all() or not np.isfinite(perf_seq).all():
        raise ValueError("non-finite pitch after MIDI conversion")

    path, distance, backend_used = compute_dtw_path(score_seq, perf_seq, cfg.backend)
    grouped = _event_ranges_from_path(path, len(score))
    pt = perf["time_s"].to_numpy(float)
    ph = perf["pitch_hz"].to_numpy(float)

    rows = []
    for i, inds in enumerate(grouped):
        row = score.iloc[i]
        if inds:
            arr = np.asarray(inds, dtype=int)
            first_j = int(arr.min())
            last_j = int(arr.max())
            med_j = int(np.median(arr))
            anchor_j = first_j if cfg.anchor_rule == "first" else med_j
            anchor_time = float(pt[anchor_j])
            anchor_pitch = float(ph[anchor_j])
            first_time = float(pt[first_j])
            median_time = float(np.median(pt[arr]))
            last_time = float(pt[last_j])
            path_count = int(len(arr))
        else:
            first_j = last_j = med_j = anchor_j = -1
            anchor_time = anchor_pitch = first_time = median_time = last_time = np.nan
            path_count = 0

        ce = cents_error(np.array([anchor_pitch]), np.array([float(row["score_pitch_hz"])]))[0]
        rows.append({
            "piece_id": piece_id,
            "method": "Pitch-only DTW",
            "score_index": int(row["score_index"]),
            "score_note_id": str(row[SCORE_ID_COL]),
            "score_time_s": float(row["score_time_s"]) if np.isfinite(row["score_time_s"]) else np.nan,
            "score_duration_s": float(row["score_duration_s"]) if np.isfinite(row["score_duration_s"]) else np.nan,
            "score_pitch_hz": float(row["score_pitch_hz"]),
            "dtw_anchor_time_s": anchor_time,
            "dtw_anchor_pitch_hz": anchor_pitch,
            "dtw_anchor_pitch_error_cents": float(ce) if np.isfinite(ce) else np.nan,
            "dtw_first_time_s": first_time,
            "dtw_median_time_s": median_time,
            "dtw_last_time_s": last_time,
            "dtw_span_ms": (last_time - first_time) * 1000.0 if np.isfinite(first_time) and np.isfinite(last_time) else np.nan,
            "dtw_path_points_for_event": path_count,
            "dtw_first_perf_index": first_j if first_j >= 0 else np.nan,
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
        n_performance_frames=len(perf),
        n_mapped_events=int(mapped.sum()),
        n_unmapped_events=int((~mapped).sum()),
        n_duplicate_anchor_times=duplicate_times,
        path_length=len(path),
        dtw_distance=distance,
        normalized_path_cost=(distance / max(1, len(path))),
        backend=backend_used,
        elapsed_s=elapsed,
    )
    return out, np.asarray(path, dtype=np.int32), perf, score, summary
