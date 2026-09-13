import os

import numpy as np
import pandas as pd


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def freq_to_midi(freq):
    freq = np.asarray(freq, dtype=float)
    freq = np.maximum(freq, 1e-9)
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_round(freq):
    return np.rint(freq_to_midi(freq)).astype(int)


def cents_error(score_f, onset_f):
    score_f = max(float(score_f), 1e-9)
    onset_f = max(float(onset_f), 1e-9)
    return 1200.0 * np.log2(onset_f / score_f)


def classify_pitch_level(cents, dropped=False):
    if dropped:
        return "F"
    a = abs(float(cents))
    if a <= 8:
        return "S"
    if a <= 20:
        return "A"
    if a <= 35:
        return "B"
    if a <= 50:
        return "C"
    if a <= 75:
        return "D"
    if a <= 100:
        return "E"
    return "F"


def classify_rhythm_level(time_ms):
    a = abs(float(time_ms))
    if a <= 80:
        return "优秀"
    if a <= 150:
        return "可接受"
    return "偏差较大"


def classify_duration_ratio_level(ratio, grace_flag):
    if not np.isfinite(ratio) or ratio < 0:
        return "F"

    if grace_flag:
        if ratio <= 3:
            return "A"
        if ratio <= 6:
            return "D"
        return "F"

    if 0.9 <= ratio <= 1.1:
        return "A"
    if (0.7 <= ratio < 0.9) or (1.1 < ratio <= 1.4):
        return "B"
    if (0.4 <= ratio < 0.7) or (1.4 < ratio <= 2.0):
        return "C"
    if (0.2 <= ratio < 0.4) or (2.0 < ratio <= 4.0):
        return "D"
    return "F"


def files_with_suffix(base_dir, suffix, case_sensitive=True):
    if case_sensitive:
        return [f for f in os.listdir(base_dir) if f.endswith(suffix)]
    suffix_l = suffix.lower()
    return [f for f in os.listdir(base_dir) if f.lower().endswith(suffix_l)]


def find_first_with_suffix(base_dir, suffix, desc=None, case_sensitive=True):
    files = files_with_suffix(base_dir, suffix, case_sensitive=case_sensitive)
    if not files:
        name = desc or f"*{suffix}"
        raise RuntimeError(f"未找到 {name} 文件")
    if len(files) > 1:
        print(f"⚠️ 检测到多个 {desc or suffix} 文件，默认使用：{files[0]}")
    return files[0]


def find_latest_with_suffix(base_dir, suffix, case_sensitive=False):
    files = files_with_suffix(base_dir, suffix, case_sensitive=case_sensitive)
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    return files[0]


def nearest_index_by_time(sorted_times, value):
    times = np.asarray(sorted_times, dtype=float)
    value = safe_float(value)
    if not np.isfinite(value) or len(times) == 0:
        return np.nan

    pos = int(np.searchsorted(times, value, side="left"))
    if pos <= 0:
        return 0.0
    if pos >= len(times):
        return float(len(times) - 1)

    prev_i = pos - 1
    next_i = pos
    if abs(times[next_i] - value) < abs(value - times[prev_i]):
        return float(next_i)
    return float(prev_i)
