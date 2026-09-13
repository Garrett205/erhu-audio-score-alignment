"""Single-source tempo-axis metadata and validation helpers.

``os.py`` is the only writer of scaled onset/pitch data.  Downstream modules
use these helpers strictly as readers/validators, never to infer another
tempo scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


RAW_TIME_COL = "时间(s)"
STRETCHED_TIME_COL = "time_stretched(s)"
ANCHOR_COL = "tempo_scale_anchor_s"
SCALE_COL = "global_scale"
SOURCE_COL = "tempo_scale_source"

ANCHOR_TOLERANCE_S = 1e-6
SCALE_TOLERANCE = 1e-9
MAPPING_TOLERANCE_S = 1e-6


def stretch_time(raw_time, anchor_s: float, global_scale: float):
    """The one approved raw-to-alignment time mapping."""
    return float(anchor_s) + (raw_time - float(anchor_s)) * float(global_scale)


def _constant_number(frame: pd.DataFrame, column: str, label: str, tolerance: float) -> float:
    if column not in frame.columns:
        raise ValueError(f"{label} 缺少时间轴元数据列: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"{label} 的时间轴元数据为空: {column}")
    first = float(values.iloc[0])
    if not math.isfinite(first) or not (values - first).abs().le(tolerance).all():
        raise ValueError(f"{label} 的时间轴元数据不是唯一有效常量: {column}")
    return first


def read_axis_metadata(frame: pd.DataFrame, label: str) -> dict[str, object]:
    """Read required, constant per-file scale metadata."""
    anchor = _constant_number(frame, ANCHOR_COL, label, ANCHOR_TOLERANCE_S)
    scale = _constant_number(frame, SCALE_COL, label, SCALE_TOLERANCE)
    if SOURCE_COL not in frame.columns:
        raise ValueError(f"{label} 缺少时间轴元数据列: {SOURCE_COL}")
    sources = frame[SOURCE_COL].dropna().astype(str).str.strip()
    sources = sources[sources.ne("")]
    if sources.empty or sources.nunique() != 1:
        raise ValueError(f"{label} 的时间轴来源缺失或不唯一: {SOURCE_COL}")
    return {ANCHOR_COL: anchor, SCALE_COL: scale, SOURCE_COL: sources.iloc[0]}


def validate_pitch_mapping(pitch_frame: pd.DataFrame, label: str = "pitch2") -> dict[str, object]:
    """Ensure the stored stretched column really follows the declared formula."""
    metadata = read_axis_metadata(pitch_frame, label)
    required = [RAW_TIME_COL, STRETCHED_TIME_COL]
    missing = [name for name in required if name not in pitch_frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少时间轴列: {missing}")
    raw = pd.to_numeric(pitch_frame[RAW_TIME_COL], errors="coerce")
    stretched = pd.to_numeric(pitch_frame[STRETCHED_TIME_COL], errors="coerce")
    usable = raw.notna() & stretched.notna()
    if not usable.any():
        raise ValueError(f"{label} 没有可验证的原始/缩放时间行")
    expected = stretch_time(raw[usable], metadata[ANCHOR_COL], metadata[SCALE_COL])
    max_error = float((expected - stretched[usable]).abs().max())
    if not math.isfinite(max_error) or max_error > MAPPING_TOLERANCE_S:
        raise ValueError("pitch2 时间轴与当前 tempo scale 不一致")
    return {**metadata, "mapping_max_error_s": max_error}


def validate_onset_pitch_axis(onset_frame: pd.DataFrame, pitch_frame: pd.DataFrame) -> dict[str, object]:
    """Reject mixed generations before TRUE extracts any pitch values."""
    onset = read_axis_metadata(onset_frame, "onset_scaled")
    pitch = validate_pitch_mapping(pitch_frame, "pitch2")
    if abs(float(onset[ANCHOR_COL]) - float(pitch[ANCHOR_COL])) > ANCHOR_TOLERANCE_S:
        raise ValueError("onset_scaled 与 pitch2 的 tempo_scale_anchor_s 不一致")
    if abs(float(onset[SCALE_COL]) - float(pitch[SCALE_COL])) > SCALE_TOLERANCE:
        raise ValueError("onset_scaled 与 pitch2 的 global_scale 不一致")
    if str(onset[SOURCE_COL]) != str(pitch[SOURCE_COL]):
        raise ValueError("onset_scaled 与 pitch2 的 tempo_scale_source 不一致")
    return pitch


def require_generated_pitch2(base_dir: str | Path, prefix: str | None = None) -> Path:
    """Find the pitch2 written by os.py without using historical formats.

    A score filename and an audio/pitch filename are legitimately allowed to
    have different prefixes.  ``os.py`` records the authoritative pitch2 name
    in tempo_scale_summary.csv, so prefer that record before applying the
    score-prefix compatibility lookup.
    """
    root = Path(base_dir)
    summary_path = root / "tempo_scale_summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            if len(summary) == 1 and "pitch2_file" in summary.columns:
                declared = str(summary.iloc[0]["pitch2_file"]).strip()
                if declared and declared.lower() != "nan":
                    declared_path = root / Path(declared).name
                    if declared_path.exists():
                        return declared_path
        except (OSError, ValueError, pd.errors.ParserError):
            # Continue to the strict filesystem fallback below.  The actual
            # metadata and mapping are still validated by TRUE after loading.
            pass

    candidates = sorted(root.glob(f"{prefix}_pitch2.csv")) if prefix else sorted(root.glob("*_pitch2.csv"))
    if not candidates:
        raise FileNotFoundError("未找到由 os.py 生成的当前 pitch2 文件，请先运行 os.py。")
    if len(candidates) != 1:
        raise RuntimeError(f"当前 pitch2 应当恰好一个，实际找到 {len(candidates)} 个: {[p.name for p in candidates]}")
    return candidates[0]
