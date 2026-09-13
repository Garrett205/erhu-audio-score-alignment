from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ONSET_ID_COL = "onset_id"
ORIGINAL_ONSET_INDEX_COL = "original_onset_index"
CURRENT_ONSET_INDEX_COL = "onset_dataframe_index"
WEAK_CANDIDATE_LEVEL = "weak"
PLATFORM_WEAK_SOURCE = "platform_transition_weak"


def ensure_onset_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach immutable source identity before sorting or filtering."""
    result = frame.copy()
    if ORIGINAL_ONSET_INDEX_COL not in result.columns:
        result[ORIGINAL_ONSET_INDEX_COL] = np.arange(len(result), dtype=int)
    original = result[ORIGINAL_ONSET_INDEX_COL]
    if original.isna().any() or original.astype(str).duplicated().any():
        raise ValueError("original_onset_index must be present and unique")
    if ONSET_ID_COL not in result.columns:
        result[ONSET_ID_COL] = [f"onset_{value}" for value in original.astype(str)]
    ids = result[ONSET_ID_COL].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("onset_id must be non-empty and unique")
    result[ONSET_ID_COL] = ids
    return result


def finalize_onset_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result[CURRENT_ONSET_INDEX_COL] = np.arange(len(result), dtype=int)
    return result


def split_normal_and_weak_onsets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep weak platform candidates out of the normal DP1/DP2 search.

    The onset table is deliberately a single, time-scaled artifact so every
    candidate shares the OS time axis.  `platform_transition_weak` rows are
    nevertheless diagnostic/local-rescue material: DP1 and DP2 must receive
    exactly the normal energy/yellow/plateau candidate set.  DP3 receives the
    second return value only for its tightly gated unmatched-gap repair.
    """
    result = frame.copy()
    level = result.get("candidate_level", pd.Series("", index=result.index))
    source = result.get("pre_src", pd.Series("", index=result.index))
    is_weak = (
        level.fillna("").astype(str).str.strip().str.lower().eq(WEAK_CANDIDATE_LEVEL)
        | source.fillna("").astype(str).str.strip().str.lower().eq(PLATFORM_WEAK_SOURCE)
    )
    return result.loc[~is_weak].copy(), result.loc[is_weak].copy()


def onset_index_by_id(frame: pd.DataFrame, onset_id: object) -> int | None:
    if ONSET_ID_COL not in frame.columns:
        return None
    key = str(onset_id).strip()
    if not key:
        return None
    matches = frame.index[frame[ONSET_ID_COL].astype(str).eq(key)].tolist()
    return int(matches[0]) if len(matches) == 1 else None


def choose_onset_file(base_dir: str | Path, prefix: str | None = None) -> Path:
    root = Path(base_dir)
    if prefix:
        for path in (root / f"{prefix}_onset_scaled.csv", root / f"{prefix}_onset_pre_note.csv"):
            if path.exists():
                return path
    scaled = sorted(root.glob("*_onset_scaled.csv"))
    raw = sorted(
        path for path in root.glob("*_onset_pre_note.csv")
        if "_pre_note_old" not in path.name and "_pre_note_gsc" not in path.name
    )
    candidates = scaled if scaled else raw
    if not candidates:
        raise FileNotFoundError("未找到 *_onset_scaled.csv 或 *_onset_pre_note.csv")
    if len(candidates) > 1:
        raise RuntimeError(f"检测到多个 onset 文件，无法安全选择: {[p.name for p in candidates]}")
    return candidates[0]
