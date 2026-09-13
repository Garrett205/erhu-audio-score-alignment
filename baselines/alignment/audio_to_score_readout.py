#!/usr/bin/env python3
"""Read-only optimistic unique-anchor diagnostic for AudioToScoreMatcher.

This is deliberately not a correspondence-accuracy metric.  It reuses the
persisted native frame path and native times, removes sentinel/negative/out of
audio-range positions, then evaluates at most one optimistic pitch success per
distinct native frame.  Matcher outputs and frozen F0 tracks are inputs only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


BASELINE_ROOT = Path(__import__("os").environ.get("ERHU_BASELINE_WORK", "work/external")).resolve()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = BASELINE_ROOT / "00_manifest/benchmark_manifest_12.csv"
ALIGNMENTS = BASELINE_ROOT / "02_alignment/audio_to_score"
NATIVE_AUDIT = BASELINE_ROOT / "09_readonly_audit/audio_native"
FROZEN_SCORE = BASELINE_ROOT / "00_manifest/frozen_score_events.csv"
F0_ROOT = Path(__import__("os").environ.get("ERHU_RUNS", "work/runs")).resolve()
CORE_SOURCE = ROOT / "baselines/alignment/pitch_onset_dtw.py"
WINDOW_S = 0.150
FRAME_STEP_S = 0.005
TOTAL_EVENTS = 1_229


def load_core():
    spec = importlib.util.spec_from_file_location("audio_lenient_frozen_core", CORE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen evaluator core: {CORE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bool_column(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise RuntimeError("Frozen hard_invalid contains an unrecognized value")
    return parsed.to_numpy(dtype=bool)


def two_frame_pitch_pass(
    core,
    time_s: np.ndarray,
    f0_hz: np.ndarray,
    valid_f0: np.ndarray,
    score_f0_hz: float,
    start_s: float,
    end_s: float,
) -> tuple[bool, int | None]:
    """Require adjacent original 5-ms frames entirely inside [start, end]."""
    indices = np.flatnonzero((time_s >= start_s - 1e-12) & (time_s <= end_s + 1e-12))
    if len(indices) < 2:
        return False, None
    cents = core.cents_error(f0_hz[indices], np.full(len(indices), score_f0_hz))
    good = valid_f0[indices] & np.isfinite(cents) & (np.abs(cents) <= 100.0)
    pairs = (
        good[:-1]
        & good[1:]
        & (np.diff(indices) == 1)
        & np.isclose(np.diff(time_s[indices]), FRAME_STEP_S, rtol=0.0, atol=1e-9)
    )
    hits = np.flatnonzero(pairs)
    return (len(hits) > 0), (int(indices[hits[0]]) if len(hits) else None)


def native_event_table(piece: str, alignment: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    native_json = ALIGNMENTS / piece / "alignment_native.json"
    positions_csv = NATIVE_AUDIT / piece / "native_positions_frames_times.csv"
    native = json.loads(native_json.read_text(encoding="utf-8"))
    native_time_by_id = {str(row["score_id"]): float(row["performance_time"]) for row in native}
    if set(native_time_by_id) != set(alignment["score_event_id"]):
        raise RuntimeError(f"{piece}: native alignment IDs do not match score events")
    positions = pd.read_csv(positions_csv)
    joined = alignment.merge(
        positions,
        left_on="score_symbolic_onset",
        right_on="score_position",
        how="left",
        validate="many_to_one",
    )
    if joined["raw_frame_index"].isna().any() or joined["raw_predicted_time"].isna().any():
        raise RuntimeError(f"{piece}: native path does not cover a score position")
    native_time = joined["score_event_id"].map(native_time_by_id).to_numpy(float)
    if not np.allclose(native_time, joined["raw_predicted_time"].to_numpy(float), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{piece}: persisted native times differ from native frame-path times")
    return joined, [native_json, positions_csv]


def evaluate_piece(manifest_row: dict[str, object], score_events: pd.DataFrame, core):
    piece = str(manifest_row["performance_id"])
    alignment_path = ALIGNMENTS / piece / "alignment.csv"
    f0_path = F0_ROOT / piece / f"{piece}_CREPE_plain_raw.csv"
    alignment = pd.read_csv(alignment_path)
    expected = score_events.loc[score_events.performance_id.eq(piece)].copy()
    if len(alignment) != int(manifest_row["event_count"]) or len(expected) != len(alignment):
        raise RuntimeError(f"{piece}: score-event population differs from frozen manifest")
    if alignment["score_event_id"].tolist() != expected["score_note_id"].tolist():
        raise RuntimeError(f"{piece}: AudioToScoreMatcher event identity differs from frozen score order")
    alignment["score_f0_hz"] = expected["score_pitch_hz"].to_numpy(float)
    native, native_paths = native_event_table(piece, alignment)

    duration_s = float(sf.info(str(manifest_row["wav_path"])).duration)
    frame = pd.read_csv(f0_path)
    time_s = pd.to_numeric(frame["时间(s)"], errors="coerce").to_numpy(float)
    f0_hz = pd.to_numeric(frame["onset频率(Hz)"], errors="coerce").to_numpy(float)
    if not np.all(np.diff(time_s) > 0):
        raise RuntimeError(f"{piece}: frozen F0 timestamps are not strictly increasing")
    valid_f0 = np.isfinite(f0_hz) & (f0_hz > 0.0) & ~bool_column(frame["hard_invalid"])

    native["raw_frame_index"] = pd.to_numeric(native["raw_frame_index"], errors="raise").astype(int)
    native["raw_predicted_time"] = pd.to_numeric(native["raw_predicted_time"], errors="raise").astype(float)
    is_valid = (
        (native["raw_frame_index"] >= 0)
        & np.isfinite(native["raw_predicted_time"])
        & (native["raw_predicted_time"] >= 0.0)
        & (native["raw_predicted_time"] <= duration_s + 1e-12)
    )
    valid_events = native.loc[is_valid].copy()
    details: list[dict[str, object]] = []
    success_count = 0
    for raw_frame, group in valid_events.groupby("raw_frame_index", sort=True):
        anchor_times = group["raw_predicted_time"].unique()
        if len(anchor_times) != 1:
            raise RuntimeError(f"{piece}: raw frame {raw_frame} has inconsistent native times")
        anchor_s = float(anchor_times[0])
        search_end_s = min(anchor_s + WINDOW_S, duration_s)
        success = False
        winner_id = None
        winner_pitch = None
        winner_frame = None
        for event in group.sort_values("score_index", kind="stable").to_dict("records"):
            passed, frame_index = two_frame_pitch_pass(
                core, time_s, f0_hz, valid_f0, float(event["score_f0_hz"]), anchor_s, search_end_s
            )
            if passed:
                success = True
                winner_id = event["score_event_id"]
                winner_pitch = int(event["score_pitch_midi"])
                winner_frame = frame_index
                break
        success_count += int(success)
        details.append(
            {
                "performance_id": piece,
                "raw_frame_index": int(raw_frame),
                "native_anchor_sec": anchor_s,
                "search_end_sec": search_end_s,
                "events_sharing_native_frame": len(group),
                "lenient_success": success,
                "winning_score_event_id": winner_id,
                "winning_score_pitch_midi": winner_pitch,
                "first_qualifying_f0_frame_index": winner_frame,
            }
        )
    distinct = len(details)
    raw_distinct = int(native["raw_frame_index"].nunique())
    summary = {
        "performance_id": piece,
        "score_events": len(native),
        "valid_outputs": len(valid_events),
        "distinct_native_anchors": distinct,
        "duplicate_outputs": len(valid_events) - distinct,
        "unique_time_coverage_pct": 100.0 * distinct / len(native),
        "lenient_unique_pass": success_count,
        "lenient_unique_pass_pct": 100.0 * success_count / len(native),
        "raw_distinct_native_frames_including_invalid": raw_distinct,
        "audio_duration_sec": duration_s,
    }
    if success_count > distinct:
        raise RuntimeError(f"{piece}: one native frame contributed more than one success")
    input_paths = [alignment_path, f0_path, *native_paths]
    return summary, details, input_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    core = load_core()
    manifest = pd.read_csv(MANIFEST)
    score_events = pd.read_csv(FROZEN_SCORE)
    if len(manifest) != 12 or int(manifest["event_count"].sum()) != TOTAL_EVENTS:
        raise RuntimeError("Expected the frozen 12-performance / 1229-event manifest")
    if len(score_events) != TOTAL_EVENTS:
        raise RuntimeError("Frozen score-event file is not 1229 events")

    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    inputs = [MANIFEST, FROZEN_SCORE, CORE_SOURCE]
    for manifest_row in manifest.to_dict("records"):
        summary, piece_details, paths = evaluate_piece(manifest_row, score_events, core)
        summaries.append(summary)
        details.extend(piece_details)
        inputs.extend(paths)
    summary_frame = pd.DataFrame(summaries)
    if int(summary_frame["score_events"].sum()) != TOTAL_EVENTS:
        raise RuntimeError("Pooled score-event denominator is not 1229")
    pooled = {
        "performance_id": "POOLED_12",
        "score_events": TOTAL_EVENTS,
        "valid_outputs": int(summary_frame["valid_outputs"].sum()),
        "distinct_native_anchors": int(summary_frame["distinct_native_anchors"].sum()),
        "duplicate_outputs": int(summary_frame["duplicate_outputs"].sum()),
        "unique_time_coverage_pct": 100.0 * summary_frame["distinct_native_anchors"].sum() / TOTAL_EVENTS,
        "lenient_unique_pass": int(summary_frame["lenient_unique_pass"].sum()),
        "lenient_unique_pass_pct": 100.0 * summary_frame["lenient_unique_pass"].sum() / TOTAL_EVENTS,
        "raw_distinct_native_frames_including_invalid": int(summary_frame["raw_distinct_native_frames_including_invalid"].sum()),
        "audio_duration_sec": np.nan,
    }
    output = pd.concat([summary_frame, pd.DataFrame([pooled])], ignore_index=True)
    output.to_csv(output_dir / "audio_to_score_lenient_unique_per_piece.csv", index=False, float_format="%.6f")
    pd.DataFrame(details).to_csv(output_dir / "audio_to_score_lenient_unique_anchor_details.csv", index=False)
    hashes = {str(path.resolve()): sha256(path) for path in sorted(set(inputs))}
    metadata = {
        "label": "optimistic unique-anchor upper-bound diagnostic",
        "method": "AudioToScoreMatcher existing native output only",
        "recordings": 12,
        "score_events": TOTAL_EVENTS,
        "native_position_unit": "raw native audio frame index; de-duplicated within each performance",
        "invalid_position_rule": "remove sentinel frame, negative time, nonfinite time, and time beyond audio end",
        "pitch_rule": "for each distinct native frame, allow any score event sharing that frame; [t,min(t+150ms,audio end)] requires two adjacent original 5-ms frozen reliability-aware F0 frames, both valid and within 100 cents",
        "matcher_rerun": False,
        "parameters_modified": False,
        "theglue_or_dualdtw_modified": False,
        "paper_modified": False,
        "xian_ju_yin": {
            "score_events": int(summary_frame.loc[summary_frame.performance_id.eq("Xian_Ju_Yin"), "score_events"].iloc[0]),
            "raw_distinct_native_frames_including_invalid": int(summary_frame.loc[summary_frame.performance_id.eq("Xian_Ju_Yin"), "raw_distinct_native_frames_including_invalid"].iloc[0]),
            "valid_distinct_native_anchors": int(summary_frame.loc[summary_frame.performance_id.eq("Xian_Ju_Yin"), "distinct_native_anchors"].iloc[0]),
        },
        "input_sha256": hashes,
    }
    (output_dir / "audio_to_score_lenient_unique_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    display_columns = [
        "performance_id", "score_events", "valid_outputs", "distinct_native_anchors",
        "duplicate_outputs", "unique_time_coverage_pct", "lenient_unique_pass",
        "lenient_unique_pass_pct",
    ]
    print(output.loc[:, display_columns].to_csv(index=False, float_format="%.6f"), end="")


if __name__ == "__main__":
    main()
