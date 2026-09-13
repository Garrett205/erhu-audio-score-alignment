from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("ERHU_WORK_DIR", Path(__file__).resolve().parent)).resolve()

TIME_COL = "time_stretched(s)"
PITCH_COL = "pitch(Hz)"
PERI_COL = "periodicity"
DB_COL = "参考强度(dB)"
HARD_INVALID_COL = "hard_invalid"

REENTRY_SOURCE = "voiced_reentry_weak"
GRACE_SOURCE = "gradual_platform_transition_weak"
TRIAL_SOURCES = {REENTRY_SOURCE, GRACE_SOURCE}

DT = 0.005
STABLE_WIN = 6
STABLE_RANGE_CENTS = 35.0
MIN_DB = 3.0
MIN_PERI = 0.03
MIN_CONFIDENCE = 0.80
MIN_INVALID_FRAMES = 16  # 80 ms
REENTRY_SUPPORT_S = 0.080  # representative pitch after the voiced entrance
EXISTING_GAP_FRAMES = 5  # 25 ms

GRACE_TARGET_CENTS = 35.0
GRACE_PLATFORM_CENTS = 70.0
GRACE_MIN_DURATION_S = 0.150
GRACE_LOOKBACK_MIN = 30   # 150 ms
GRACE_LOOKBACK_MAX = 160  # 800 ms
GRACE_MONOTONIC_RATIO = 0.75


def find_one(pattern: str, root: Path = WORK) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {pattern} in {root}, found {[p.name for p in hits]}")
    return hits[0]


def cents_error(reference_hz: float, observed_hz: float) -> float:
    if not (np.isfinite(reference_hz) and np.isfinite(observed_hz)) or reference_hz <= 0 or observed_hz <= 0:
        return np.nan
    return float(1200.0 * np.log2(observed_hz / reference_hz))


def robust_range_cents(values: np.ndarray) -> float:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size < 2:
        return np.inf
    low, high = np.percentile(valid, [10, 90])
    return float(1200.0 * np.log2((high + 1e-9) / (low + 1e-9)))


def stable_stats(freq: np.ndarray, db: np.ndarray, peri: np.ndarray, start: int):
    end = start + STABLE_WIN
    if start < 0 or end > len(freq):
        return None
    window = freq[start:end]
    valid = window[np.isfinite(window) & (window > 0)]
    valid_ratio = float(valid.size / len(window))
    if valid.size < 2 or valid_ratio < 0.80:
        return None
    pitch_range = robust_range_cents(valid)
    mean_db = float(np.nanmean(db[start:end]))
    mean_peri = float(np.nanmean(peri[start:end]))
    if pitch_range > STABLE_RANGE_CENTS or mean_db < MIN_DB or mean_peri < MIN_PERI:
        return None
    return {
        "median_f0": float(np.median(valid)),
        "valid_ratio": valid_ratio,
        "range_cents": pitch_range,
        "mean_db": mean_db,
        "mean_peri": mean_peri,
    }


def candidate_confidence(stats: dict, support: float) -> float:
    stability = 1.0 - min(1.0, stats["range_cents"] / STABLE_RANGE_CENTS)
    return float(np.mean([
        stats["valid_ratio"],
        stability,
        min(1.0, stats["mean_db"] / 6.0),
        min(1.0, stats["mean_peri"] / 0.10),
        min(1.0, max(0.0, support)),
    ]))


def find_reentry_candidates(
    times: np.ndarray,
    freq: np.ndarray,
    db: np.ndarray,
    peri: np.ndarray,
    hard_invalid: np.ndarray,
    existing_frames: set[int],
):
    invalid = hard_invalid | ~np.isfinite(freq) | (freq <= 0)
    candidates = []
    for start in range(1, len(freq) - STABLE_WIN):
        if invalid[start] or not invalid[start - 1]:
            continue
        invalid_start = start - 1
        while invalid_start >= 0 and invalid[invalid_start]:
            invalid_start -= 1
        invalid_run = start - invalid_start - 1
        if invalid_run < MIN_INVALID_FRAMES:
            continue
        if any(abs(start - frame) < EXISTING_GAP_FRAMES for frame in existing_frames):
            continue
        stats = stable_stats(freq, db, peri, start)
        if stats is None:
            continue
        confidence = candidate_confidence(stats, invalid_run / float(2 * MIN_INVALID_FRAMES))
        if confidence < MIN_CONFIDENCE:
            continue
        support_end = int(np.searchsorted(times, times[start] + REENTRY_SUPPORT_S, side="right"))
        support = freq[start:max(start + STABLE_WIN, support_end)]
        support = support[np.isfinite(support) & (support > 0)]
        if support.size < STABLE_WIN:
            continue
        support_f0 = float(np.median(support))
        candidates.append({
            "note_frame": start,
            "note_time(s)": float(times[start]),
            "note_f0(Hz)": support_f0,
            "pre_src": REENTRY_SOURCE,
            "platform_left_f0": 0.0,
            "platform_right_f0": support_f0,
            "platform_diff_cents": np.nan,
            "platform_local_change_cents": np.nan,
            "candidate_confidence": confidence,
        })
    return candidates


def compatible_clusters(times: np.ndarray, freq: np.ndarray, score_hz: float, left: float, right: float):
    cents = np.full(len(freq), np.nan, dtype=float)
    valid = np.isfinite(freq) & (freq > 0)
    cents[valid] = 1200.0 * np.log2(freq[valid] / score_hz)
    mask = valid & (np.abs(cents) <= GRACE_TARGET_CENTS) & (times > left) & (times < right)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [group for group in groups if group.size >= 2]


def find_grace_main_candidates(
    align: pd.DataFrame,
    times: np.ndarray,
    freq: np.ndarray,
    db: np.ndarray,
    peri: np.ndarray,
    existing_frames: set[int],
):
    candidates = []
    success = align["对齐状态"].astype(str).eq("对齐成功").to_numpy()
    for index in range(1, len(align)):
        row = align.iloc[index]
        previous = align.iloc[index - 1]
        if success[index] or not success[index - 1]:
            continue
        if str(row.get("装饰音判定", "")).strip() != "主音":
            continue
        if str(previous.get("装饰音判定", "")).strip() != "装饰音":
            continue
        previous_score_time = pd.to_numeric(pd.Series([previous.get("谱面时间_原始(s)")]), errors="coerce").iloc[0]
        current_score_time = pd.to_numeric(pd.Series([row.get("谱面时间_原始(s)")]), errors="coerce").iloc[0]
        if not (np.isfinite(previous_score_time) and np.isfinite(current_score_time)):
            continue
        if abs(float(previous_score_time) - float(current_score_time)) > 1e-9:
            continue

        left_time = float(previous["红点时间(s)"])
        right_time = np.inf
        for right_index in range(index + 1, len(align)):
            if success[right_index]:
                right_time = float(align.iloc[right_index]["红点时间(s)"])
                break
        score_hz = float(row["谱面频率(Hz)"])
        previous_score_hz = float(previous["谱面频率(Hz)"])

        for group in compatible_clusters(times, freq, score_hz, left_time, right_time):
            start = int(group[0])
            end = int(group[-1])
            if times[end] - times[start] < GRACE_MIN_DURATION_S:
                continue
            if any(abs(start - frame) < EXISTING_GAP_FRAMES for frame in existing_frames):
                continue
            right_stats = stable_stats(freq, db, peri, start)
            if right_stats is None:
                continue

            best_left = None
            for left_end in range(start - GRACE_LOOKBACK_MIN, max(STABLE_WIN, start - GRACE_LOOKBACK_MAX) - 1, -1):
                left_start = left_end - STABLE_WIN
                left_stats = stable_stats(freq, db, peri, left_start)
                if left_stats is None:
                    continue
                left_pitch_error = cents_error(previous_score_hz, left_stats["median_f0"])
                if not np.isfinite(left_pitch_error) or abs(left_pitch_error) > GRACE_PLATFORM_CENTS:
                    continue
                transition = freq[left_end:start + 1]
                transition = transition[np.isfinite(transition) & (transition > 0)]
                if transition.size < 3:
                    continue
                signed_total = cents_error(left_stats["median_f0"], right_stats["median_f0"])
                if not np.isfinite(signed_total) or abs(signed_total) < 70.0:
                    continue
                steps = 1200.0 * np.diff(np.log2(transition + 1e-9))
                direction = 1.0 if signed_total > 0 else -1.0
                monotonic_ratio = float(np.mean(direction * steps >= -2.0))
                if monotonic_ratio < GRACE_MONOTONIC_RATIO:
                    continue
                best_left = (left_stats, signed_total, monotonic_ratio, steps)
                break
            if best_left is None:
                continue

            left_stats, signed_total, monotonic_ratio, steps = best_left
            confidence = float(np.mean([
                candidate_confidence(right_stats, 1.0),
                candidate_confidence(left_stats, 1.0),
                monotonic_ratio,
                min(1.0, abs(signed_total) / 140.0),
            ]))
            if confidence < MIN_CONFIDENCE:
                continue
            candidates.append({
                "note_frame": start,
                "note_time(s)": float(times[start]),
                "note_f0(Hz)": float(np.nanmean(freq[start:start + 4])),
                "pre_src": GRACE_SOURCE,
                "platform_left_f0": left_stats["median_f0"],
                "platform_right_f0": right_stats["median_f0"],
                "platform_diff_cents": abs(signed_total),
                "platform_local_change_cents": float(np.max(np.abs(steps))) if steps.size else 0.0,
                "candidate_confidence": confidence,
                "target_score_note_id": str(row.get("score_note_id", "")),
            })
            break
    return candidates


def make_onset_row(template_columns, candidate: dict, original_index: int, metadata: dict):
    row = {column: np.nan for column in template_columns}
    row.update(metadata)
    row.update({
        "note_time(s)": candidate["note_time(s)"],
        "note_f0(Hz)": candidate["note_f0(Hz)"],
        "note_note(cent)": "",
        "note_frame": int(candidate["note_frame"]),
        "note_color": "red",
        "red_pitch_source": "pitch2_structured_weak",
        "pre_time(s)": candidate["note_time(s)"],
        "pre_frame": int(candidate["note_frame"]),
        "pre_src": candidate["pre_src"],
        "pre_color": "purple",
        "candidate_level": "weak",
        "platform_left_f0": candidate["platform_left_f0"],
        "platform_right_f0": candidate["platform_right_f0"],
        "platform_diff_cents": candidate["platform_diff_cents"],
        "platform_local_change_cents": candidate["platform_local_change_cents"],
        "candidate_confidence": candidate["candidate_confidence"],
        "stable_start_time(s)": candidate["note_time(s)"],
        "pre_to_note_ms": 0.0,
        "original_onset_index": original_index,
        "onset_id": f"onset_{original_index}",
        "onset_dataframe_index": original_index,
    })
    return row


def main() -> int:
    case_id = WORK.name
    onset_path = find_one("*_onset_scaled.csv")
    pitch_path = find_one("*_pitch2.csv")
    baseline_dir = ROOT / "baseline_results" / case_id
    align_hits = sorted(baseline_dir.glob("*DP3_对齐结果_score主导.csv"))
    if len(align_hits) != 1:
        raise RuntimeError(f"missing unique baseline DP3 alignment for {case_id}")

    onset = pd.read_csv(onset_path)
    pitch = pd.read_csv(pitch_path)
    align = pd.read_csv(align_hits[0])
    if "pre_src" in onset.columns:
        onset = onset.loc[~onset["pre_src"].fillna("").astype(str).isin(TRIAL_SOURCES)].copy()

    times = pd.to_numeric(pitch[TIME_COL], errors="coerce").to_numpy(dtype=float)
    freq = pd.to_numeric(pitch[PITCH_COL], errors="coerce").to_numpy(dtype=float)
    peri = pd.to_numeric(pitch.get(PERI_COL, pd.Series(1.0, index=pitch.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    db = pd.to_numeric(pitch.get(DB_COL, pd.Series(0.0, index=pitch.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hard_invalid = (
        pitch.get(HARD_INVALID_COL, pd.Series(False, index=pitch.index))
        .fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        .to_numpy(dtype=bool)
    )
    existing_frames = {
        int(value)
        for value in pd.to_numeric(onset.get("note_frame", pd.Series(dtype=float)), errors="coerce").dropna()
        if int(value) >= 0
    }

    candidates = find_reentry_candidates(times, freq, db, peri, hard_invalid, existing_frames)
    candidates.extend(find_grace_main_candidates(align, times, freq, db, peri, existing_frames))

    # Keep one candidate per physical frame/source, highest confidence first.
    deduplicated = {}
    for candidate in candidates:
        key = (int(candidate["note_frame"]), str(candidate["pre_src"]))
        previous = deduplicated.get(key)
        if previous is None or candidate["candidate_confidence"] > previous["candidate_confidence"]:
            deduplicated[key] = candidate
    candidates = sorted(deduplicated.values(), key=lambda item: (item["note_time(s)"], item["pre_src"]))

    original_numeric = pd.to_numeric(onset.get("original_onset_index", pd.Series(dtype=float)), errors="coerce")
    next_index = int(original_numeric.max()) + 1 if original_numeric.notna().any() else len(onset)
    metadata = {}
    for column in ["tempo_scale_source", "tempo_scale_factor", "tempo_scale_anchor_s", "global_scale"]:
        if column in onset.columns and len(onset):
            metadata[column] = onset.iloc[0][column]
    new_rows = []
    for offset, candidate in enumerate(candidates):
        new_rows.append(make_onset_row(onset.columns, candidate, next_index + offset, metadata))
    if new_rows:
        onset = pd.concat([onset, pd.DataFrame(new_rows, columns=onset.columns)], ignore_index=True)

    temp_path = onset_path.with_suffix(onset_path.suffix + ".trial_tmp")
    onset.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, onset_path)

    diagnostic = WORK / f"{case_id}_structured_weak_trial.csv"
    pd.DataFrame(candidates).to_csv(diagnostic, index=False, encoding="utf-8-sig")
    counts = pd.Series([item["pre_src"] for item in candidates], dtype=str).value_counts().to_dict()
    print(f"{case_id}: appended={len(candidates)} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
