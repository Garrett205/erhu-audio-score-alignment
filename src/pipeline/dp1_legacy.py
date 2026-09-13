import os
import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill

from common_utils import (
    cents_error,
    classify_duration_ratio_level,
    classify_pitch_level,
    classify_rhythm_level,
    freq_to_midi,
    find_first_with_suffix,
    midi_round,
    safe_numeric,
)
from alignment_contract import (
    CURRENT_ONSET_INDEX_COL,
    ONSET_ID_COL,
    ORIGINAL_ONSET_INDEX_COL,
    choose_onset_file,
    ensure_onset_identity,
    finalize_onset_dataframe,
    split_normal_and_weak_onsets,
)

# =========================
# 全局配置
# =========================
RUN_TAG = "DP1"
METHOD_TAG_CN = "score主导"

# ===== 主DP参数 =====
SKIP_ONSET_PENALTY = 0.8
PROGRESS_WEIGHT    = 1.5
SEMITONE_SOFT_COST = 1.2

# ===== 硬前卷 earliest =====
LOOKBACK_ONSET_NUM   = 10
HARD_EARLY_EQ_CENTS  = 50
ENABLE_ONSET_DEDUP   = True

# ===== 坏区删除规则 =====
ENABLE_BAD_REGION_DROP = True
BAD_REGION_GAP         = 2   # 原始可疑点之间允许 gap<=2 合并
BAD_REGION_MIN_SUS     = 2   # 至少 2 个原始可疑点才形成坏区间
BAD_REGION_EXTEND      = 1   # 坏区间左右各延伸 1 个音

# ===== 列名（score）=====
SCORE_TIME_COL   = "累计时间(s)"
SCORE_F0_COL     = "音高频率(Hz)"
SCORE_NAME_COL   = "音名"
SCORE_DUR_COL    = "持续时间(s)"
SCORE_BPM_COL    = "BPM"
SCORE_MISC_COL   = "连音线和其他信息"
SCORE_LEGATO_COL = "连奏信息"
SCORE_GRACE_COL  = "装饰音判定"

# ===== 列名（onset）=====
ONSET_F0_COL = "note_f0(Hz)"
ONSET_T_COL  = "note_time(s)"


def pitch_match_cost(score_midi, onset_midi):
    sem = abs(int(score_midi) - int(onset_midi))
    if sem == 0:
        return 0.0
    if sem == 1:
        return SEMITONE_SOFT_COST
    return 4.0 + 2.0 * (sem - 2)


def main_fine_pitch_penalty(score_f, onset_f_used):
    a = abs(cents_error(score_f, onset_f_used))

    if a <= 25:
        return 0.03 * (a / 25.0)
    elif a <= 50:
        return 0.03 + 0.07 * ((a - 25) / 25.0)
    elif a <= 100:
        return 0.10 + 0.20 * ((a - 50) / 50.0)
    else:
        return 0.30 + 2.20 * ((a - 100) / 100.0)


# ============================================================
# 文件读取
# ============================================================
def find_score_file(base_dir):
    return find_first_with_suffix(base_dir, "_score_final.csv", "*_score_final.csv")


def find_onset_file(base_dir):
    return choose_onset_file(base_dir).name


def load_data(base_dir):
    score_file = find_score_file(base_dir)
    onset_file = find_onset_file(base_dir)

    df_sc = pd.read_csv(os.path.join(base_dir, score_file))
    df_on = ensure_onset_identity(pd.read_csv(os.path.join(base_dir, onset_file)))
    df_on, _weak_onsets = split_normal_and_weak_onsets(df_on)

    for c in [SCORE_TIME_COL, SCORE_F0_COL, SCORE_DUR_COL]:
        if c not in df_sc.columns:
            raise RuntimeError(f"score 文件缺少列: {c}")
    for c in [ONSET_T_COL, ONSET_F0_COL]:
        if c not in df_on.columns:
            raise RuntimeError(f"onset 文件缺少列: {c}")

    df_sc[SCORE_TIME_COL] = safe_numeric(df_sc[SCORE_TIME_COL])
    df_sc[SCORE_F0_COL] = safe_numeric(df_sc[SCORE_F0_COL])
    df_sc[SCORE_DUR_COL] = safe_numeric(df_sc[SCORE_DUR_COL])

    df_on[ONSET_T_COL] = safe_numeric(df_on[ONSET_T_COL])
    df_on[ONSET_F0_COL] = safe_numeric(df_on[ONSET_F0_COL])

    df_sc = df_sc.dropna(subset=[SCORE_TIME_COL, SCORE_F0_COL]).reset_index(drop=True)
    df_on = df_on.dropna(subset=[ONSET_T_COL, ONSET_F0_COL]).reset_index(drop=True)

    df_on = df_on.sort_values(ONSET_T_COL).reset_index(drop=True)
    if ENABLE_ONSET_DEDUP:
        df_on["_t_round"] = df_on[ONSET_T_COL].round(4)
        before = len(df_on)
        df_on = (
            df_on.drop_duplicates(subset=["_t_round"], keep="first")
            .drop(columns=["_t_round"])
            .reset_index(drop=True)
        )
        after = len(df_on)
        print(f"🧹 onset 时间去重: {before} -> {after}")

    df_on = finalize_onset_dataframe(df_on)

    prefix = os.path.splitext(score_file)[0].replace("_score_final", "")
    return df_sc, df_on, prefix, score_file, onset_file


def get_score_time_array(df_sc):
    arr = safe_numeric(df_sc[SCORE_TIME_COL]).to_numpy(dtype=float)
    return arr if np.any(np.isfinite(arr)) else None


def get_onset_time_array(df_on):
    return safe_numeric(df_on[ONSET_T_COL]).to_numpy(dtype=float)


# ============================================================
# 硬前卷 earliest refine（纯原频率版本）
# ============================================================
def hard_earliest_rewind_one(
    sc_idx,
    cur_on_idx,
    prev_final_on_idx,
    sc_m,
    on_m,
    sc_f,
    on_f,
    eq_cents=75.0,
    lookback_num=10,
):
    """
    从当前 onset 向前连续回溯：
    - 每一个 onset 都必须满足 |cents| <= eq_cents
    - 一旦某个不满足，立刻停止
    - 返回最前面的连续合格 onset
    """
    left_bound = max(prev_final_on_idx + 1, cur_on_idx - lookback_num)
    if left_bound >= cur_on_idx:
        return cur_on_idx, "无可回溯空间"

    best = cur_on_idx
    for cand in range(cur_on_idx, left_bound - 1, -1):
        c = abs(cents_error(sc_f[sc_idx], on_f[cand]))
        if c <= eq_cents:
            best = cand
        else:
            break

    return best, "连续阈值前卷"


def apply_hard_earliest_refine(pairs, sc_m, on_m, sc_f, on_f, eq_cents=75.0, lookback_num=10):
    if not pairs:
        return pairs

    refined = []
    prev_final_on_idx = -1

    for item in pairs:
        sc_idx, on_idx, _, _ = item

        new_on_idx, _ = hard_earliest_rewind_one(
            sc_idx=sc_idx,
            cur_on_idx=on_idx,
            prev_final_on_idx=prev_final_on_idx,
            sc_m=sc_m,
            on_m=on_m,
            sc_f=sc_f,
            on_f=on_f,
            eq_cents=eq_cents,
            lookback_num=lookback_num,
        )

        refined.append((sc_idx, new_on_idx, "否", False))
        prev_final_on_idx = new_on_idx

    return refined


# ============================================================
# 主匹配：全局DP（纯原频率版本）
# ============================================================
def build_dp_alignment(sc_m, on_m, sc_f, on_f, score_t=None, onset_t=None):
    n = len(sc_m)
    m = len(on_m)
    INF = 1e15

    dp = np.full((n + 1, m + 1), INF, dtype=float)
    prev = np.empty((n + 1, m + 1), dtype=object)

    dp[0, :] = np.arange(m + 1) * SKIP_ONSET_PENALTY
    prev[0, 0] = None
    for j in range(1, m + 1):
        prev[0, j] = ("skip_onset", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i, j - 1] + SKIP_ONSET_PENALTY
            best_prev = ("skip_onset", i, j - 1)

            pcost = pitch_match_cost(sc_m[i - 1], on_m[j - 1])
            onset_f_used = on_f[j - 1]
            fine_cost = main_fine_pitch_penalty(sc_f[i - 1], onset_f_used)
            progress_cost = abs(j / m - i / n) * PROGRESS_WEIGHT

            cand = dp[i - 1, j - 1] + pcost + fine_cost + progress_cost

            if cand < best:
                best = cand
                best_prev = ("match", i - 1, j - 1, "否", False)

            dp[i, j] = best
            prev[i, j] = best_prev

    j_best = int(np.argmin(dp[n, :]))

    pairs = []
    i, j = n, j_best
    while i > 0:
        action = prev[i, j]
        if action is None:
            break

        if action[0] == "skip_onset":
            _, i2, j2 = action
            i, j = i2, j2
        else:
            _, sc_idx, on_idx, octave_flag, use_octave_fix = action
            pairs.append((sc_idx, on_idx, octave_flag, use_octave_fix))
            i, j = sc_idx, on_idx

    pairs.reverse()
    return pairs


# ============================================================
# 输出结果表
# ============================================================
def build_align_table(df_sc, df_on, pairs):
    temp_rows = []

    for sc_idx, on_idx, octave_flag, use_octave_fix in pairs:
        sc = df_sc.loc[sc_idx]
        on = df_on.loc[on_idx]

        score_time_raw = float(sc[SCORE_TIME_COL]) if pd.notna(sc[SCORE_TIME_COL]) else np.nan
        score_f = float(sc[SCORE_F0_COL])

        red_t = float(on[ONSET_T_COL])
        red_f_raw = float(on[ONSET_F0_COL])

        red_f_used = red_f_raw
        low_oct_fix = "否"
        octave_flag = "否"

        raw_time_error_s = red_t - score_time_raw if np.isfinite(score_time_raw) else np.nan
        pitch_err_cents = cents_error(score_f, red_f_used)

        temp_rows.append(
            {
                "score_time_raw": score_time_raw,
                "score_f": score_f,
                "red_t": red_t,
                "red_f_raw": red_f_raw,
                "red_f_used": red_f_used,
                ONSET_ID_COL: on[ONSET_ID_COL],
                ORIGINAL_ONSET_INDEX_COL: on[ORIGINAL_ONSET_INDEX_COL],
                CURRENT_ONSET_INDEX_COL: on[CURRENT_ONSET_INDEX_COL],
                "low_oct_fix": low_oct_fix,
                "octave_flag": octave_flag,
                "raw_time_error_s": raw_time_error_s,
                "pitch_err_cents": pitch_err_cents,
                "音名": sc[SCORE_NAME_COL] if SCORE_NAME_COL in df_sc.columns else "",
                "持续时间(s)": float(sc[SCORE_DUR_COL])
                if SCORE_DUR_COL in df_sc.columns and pd.notna(sc[SCORE_DUR_COL])
                else np.nan,
                "累计时长(s)": score_time_raw,
                "连音线和其他信息": sc[SCORE_MISC_COL] if SCORE_MISC_COL in df_sc.columns else "",
                "连奏信息": sc[SCORE_LEGATO_COL] if SCORE_LEGATO_COL in df_sc.columns else "",
                "装饰音判定": sc[SCORE_GRACE_COL] if SCORE_GRACE_COL in df_sc.columns else "",
                "BPM": float(sc[SCORE_BPM_COL])
                if SCORE_BPM_COL in df_sc.columns and pd.notna(sc[SCORE_BPM_COL])
                else np.nan,
            }
        )

    df_temp = pd.DataFrame(temp_rows)
    score_shift = float(df_temp["raw_time_error_s"].mean()) if len(df_temp) > 0 else 0.0

    rows = []
    for _, r in df_temp.iterrows():
        score_time_shifted = r["score_time_raw"] + score_shift if np.isfinite(r["score_time_raw"]) else np.nan
        time_err_ms = (r["red_t"] - score_time_shifted) * 1000.0 if np.isfinite(score_time_shifted) else np.nan
        pitch_err_cents = r["pitch_err_cents"]

        rows.append(
            {
                "谱面时间_原始(s)": r["score_time_raw"],
                "谱面时间_平移后(s)": score_time_shifted,
                "score_shift(s)": score_shift,
                "谱面频率(Hz)": r["score_f"],
                "红点时间(s)": r["red_t"],
                "红点频率_raw(Hz)": r["red_f_raw"],
                "红点频率_used(Hz)": r["red_f_used"],
                ONSET_ID_COL: r[ONSET_ID_COL],
                ORIGINAL_ONSET_INDEX_COL: r[ORIGINAL_ONSET_INDEX_COL],
                CURRENT_ONSET_INDEX_COL: r[CURRENT_ONSET_INDEX_COL],
                "低音八度修正": r["low_oct_fix"],
                "高八度等效匹配": r["octave_flag"],
                "时间误差(ms)": time_err_ms,
                "音准误差(cents)": pitch_err_cents,
                "窗口(s)": np.nan,
                "对齐状态": "对齐成功",
                "后删标记": "",
                "音准等级": classify_pitch_level(pitch_err_cents),
                "节奏等级": classify_rhythm_level(time_err_ms),
                "音名": r["音名"],
                "持续时间(s)": r["持续时间(s)"],
                "累计时长(s)": r["累计时长(s)"],
                "连音线和其他信息": r["连音线和其他信息"],
                "连奏信息": r["连奏信息"],
                "装饰音判定": r["装饰音判定"],
                "BPM": r["BPM"],
            }
        )

    return pd.DataFrame(rows)


def append_duration_ratio_columns(df_align):
    if len(df_align) == 0:
        return df_align

    actual_gaps = []
    score_gaps = []
    ratios = []
    ratio_pcts = []
    ratio_levels = []

    if "谱面时间_平移后(s)" in df_align.columns:
        score_times = pd.to_numeric(
            df_align["谱面时间_平移后(s)"],
            errors="coerce",
        ).to_numpy(dtype=float)
    else:
        score_times = pd.to_numeric(
            df_align["谱面时间_原始(s)"],
            errors="coerce",
        ).to_numpy(dtype=float)
    onset_times = pd.to_numeric(df_align["红点时间(s)"], errors="coerce").to_numpy(dtype=float)
    score_durs = pd.to_numeric(df_align["持续时间(s)"], errors="coerce").to_numpy(dtype=float)

    if "装饰音判定" in df_align.columns:
        grace_flags = df_align["装饰音判定"].fillna("").astype(str).to_list()
    else:
        grace_flags = [""] * len(df_align)

    if "BPM" in df_align.columns:
        bpm_vals = pd.to_numeric(df_align["BPM"], errors="coerce").to_numpy(dtype=float)
    else:
        bpm_vals = np.full(len(df_align), np.nan, dtype=float)

    last_black_time = np.nan
    if len(df_align) > 0 and np.isfinite(score_times[-1]) and np.isfinite(score_durs[-1]):
        last_black_time = score_times[-1] + score_durs[-1]

    for i in range(len(df_align)):
        if i < len(df_align) - 1:
            actual_gap = onset_times[i + 1] - onset_times[i]
        else:
            actual_gap = last_black_time - onset_times[i] if np.isfinite(last_black_time) else np.nan

        is_grace = grace_flags[i].strip() == "装饰音"

        if is_grace:
            bpm_val = bpm_vals[i]
            score_gap = 6.0 / bpm_val if np.isfinite(bpm_val) and bpm_val > 0 else np.nan
        else:
            if i < len(df_align) - 1:
                score_gap = score_times[i + 1] - score_times[i]
            else:
                score_gap = last_black_time - score_times[i] if np.isfinite(last_black_time) else np.nan

        if not np.isfinite(actual_gap) or not np.isfinite(score_gap) or score_gap <= 0:
            ratio = np.nan
            ratio_pct = np.nan
            ratio_level = "F"
        else:
            ratio = actual_gap / score_gap
            ratio_pct = (ratio - 1.0) * 100.0
            ratio_level = classify_duration_ratio_level(ratio, is_grace)

        actual_gaps.append(actual_gap)
        score_gaps.append(score_gap)
        ratios.append(ratio)
        ratio_pcts.append(ratio_pct)
        ratio_levels.append(ratio_level)

    df_align["实际推进时长(s)"] = actual_gaps
    df_align["理论推进时长(s)"] = score_gaps
    df_align["时值比"] = ratios
    df_align["时值偏差率(%)"] = ratio_pcts
    df_align["时值等级"] = ratio_levels
    return df_align


# ============================================================
# 可疑点 / 坏区间
# ============================================================
def mark_suspicious_and_bad_regions(df_align, max_gap=2, min_sus_points=2, extend=1):
    """
    规则：
    1) 原始可疑点(seed):
       - 音准F 或 时值F
    2) 连带可疑点(display only):
       - 如果某点时值F，则下一个点也标为可疑
       - 但该“连带可疑点”不参与坏区间成段
    3) 坏区间只用原始可疑点 seed 按 gap<=2 合并
    4) 组内至少 2 个原始可疑点 -> 形成坏区间
    5) 坏区间左右各延伸 1 个音
    6) 删除范围内：
       - 对齐状态 = 未匹配(坏区删除)
       - 坏点标记 = 坏点
       - 可疑坏区间标注 = 坏区...
    7) 单点可疑但未进入坏区间的，保留可疑标记，不删除
    """
    df = df_align.copy()
    n = len(df)

    df["可疑点标记"] = ""
    df["坏点标记"] = ""
    df["可疑坏区间标注"] = ""

    if n == 0:
        return df, []

    pitch_levels = df["音准等级"].fillna("").astype(str).to_numpy()
    dur_levels = df["时值等级"].fillna("").astype(str).to_numpy()

    seed_mask = np.array(
        [(pitch_levels[i] == "F") or (dur_levels[i] == "F") for i in range(n)],
        dtype=bool,
    )

    display_mask = seed_mask.copy()

    for i in range(n):
        if dur_levels[i] == "F":
           if i - 1 >= 0:
              display_mask[i - 1] = True
           if i + 1 < n:
              display_mask[i + 1] = True

    df.loc[seed_mask, "可疑点标记"] = "可疑点"
    for i in range(n):
        if display_mask[i] and not seed_mask[i]:
            df.at[i, "可疑点标记"] = "连带可疑点"
            if df.at[i, "可疑坏区间标注"] == "":
                df.at[i, "可疑坏区间标注"] = "前一时值F连带"

    suspicious_idx = np.where(seed_mask)[0].tolist()
    if not suspicious_idx:
        return df, []

    groups = []
    cur = [suspicious_idx[0]]
    for idx in suspicious_idx[1:]:
        gap = idx - cur[-1] - 1
        if gap <= max_gap:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)

    regions = []
    region_id = 0

    for group in groups:
        if len(group) < min_sus_points:
            for idx in group:
                if df.at[idx, "可疑坏区间标注"] == "":
                    df.at[idx, "可疑坏区间标注"] = "单点可疑"
            continue

        region_id += 1
        raw_l = group[0]
        raw_r = group[-1]
        drop_l = max(0, raw_l - extend)
        drop_r = min(n - 1, raw_r + extend)

        tag = (
            f"坏区{region_id}[原可疑:{raw_l}-{raw_r}; "
            f"删除:{drop_l}-{drop_r}; 原始可疑点数:{len(group)}]"
        )

        df.loc[drop_l:drop_r, "坏点标记"] = "坏点"
        df.loc[drop_l:drop_r, "可疑坏区间标注"] = tag
        df.loc[drop_l:drop_r, "对齐状态"] = "未匹配(坏区删除)"
        df.loc[drop_l:drop_r, "后删标记"] = "是"

        regions.append(
            {
                "region_id": region_id,
                "raw_suspicious_start": raw_l,
                "raw_suspicious_end": raw_r,
                "drop_start": drop_l,
                "drop_end": drop_r,
                "suspicious_count": len(group),
            }
        )

    return df, regions


def build_metrics_table(df_align):
    total = len(df_align)

    status_ok = df_align["对齐状态"].astype(str).str.strip().eq("对齐成功")
    pitch_err_all = pd.to_numeric(
        df_align["音准误差(cents)"], errors="coerce"
    )

    unmatched_mask = ~status_ok
    wrong_pitch_mask = status_ok & pitch_err_all.abs().gt(100.0)
    # B = unmatched/conflict/deleted OR provisional match with |pitch error| > 100 cents.
    # This is reporting-only and must never modify df_align or its alignment statuses.
    b_error_mask = unmatched_mask | wrong_pitch_mask
    eval_valid_mask = ~b_error_mask
    df_valid = df_align[eval_valid_mask].copy()

    # Retain legacy status-only final-match fields for downstream compatibility.
    legacy_success = int(status_ok.sum())
    success = int(eval_valid_mask.sum())
    dropped = int(unmatched_mask.sum())
    wrong_pitch_count = int(wrong_pitch_mask.sum())
    b_error_count = int(b_error_mask.sum())
    suspicious_count = int(
        (df_align["可疑点标记"].isin(["可疑点", "连带可疑点"])).sum()
    ) if "可疑点标记" in df_align.columns else 0
    bad_count = int((df_align["坏点标记"] == "坏点").sum()) if "坏点标记" in df_align.columns else 0
    legacy_success_rate = legacy_success / total * 100.0 if total > 0 else np.nan
    b_success_rate = success / total * 100.0 if total > 0 else np.nan

    time_err = safe_numeric(df_valid["时间误差(ms)"]) if success > 0 else pd.Series(dtype=float)
    pitch_err = safe_numeric(df_valid["音准误差(cents)"]) if success > 0 else pd.Series(dtype=float)

    pitch_counts = df_valid["音准等级"].value_counts() if success > 0 else pd.Series(dtype=int)
    rhythm_counts = df_valid["节奏等级"].value_counts() if success > 0 else pd.Series(dtype=int)
    dur_counts = df_valid["时值等级"].value_counts() if success > 0 else pd.Series(dtype=int)

    # 为兼容后续流程保留，但现在恒为 0
    octave_fix_count = 0
    octave_match_count = 0
    octave_hold_count = 0

    score_shift_val = round(float(df_align["score_shift(s)"].iloc[0]) if total > 0 else 0.0, 6)

    return pd.DataFrame(
        [
            {
                "总谱面音数": total,
                "可疑点总数(含连带)": suspicious_count,
                "坏点总数(删除范围内)": bad_count,
                "删除未匹配数": dropped,
                "错音高>100c数": wrong_pitch_count,
                "B类错误数(错音高或未匹配)": b_error_count,
                "B口径有效对应数": success,
                "B口径有效对应率(%)": round(b_success_rate, 2),
                "最终匹配成功数": legacy_success,
                "最终匹配率(%)": round(legacy_success_rate, 2),
                "score_shift(s)": score_shift_val,
                "平均时间误差(ms)": round(time_err.mean(), 2) if success > 0 else np.nan,
                "时间误差中位数(ms)": round(time_err.median(), 2) if success > 0 else np.nan,
                "时间误差标准差(ms)": round(time_err.std(), 2) if success > 0 else np.nan,
                "平均绝对时间误差(ms)": round(time_err.abs().mean(), 2) if success > 0 else np.nan,
                "平均音准误差(cents)": round(pitch_err.mean(), 2) if success > 0 else np.nan,
                "音准误差中位数(cents)": round(pitch_err.median(), 2) if success > 0 else np.nan,
                "音准误差标准差(cents)": round(pitch_err.std(), 2) if success > 0 else np.nan,
                "平均绝对音准误差(cents)": round(pitch_err.abs().mean(), 2) if success > 0 else np.nan,
                "音准S(<=8c)个数": int(pitch_counts.get("S", 0)),
                "音准A(<=20c)个数": int(pitch_counts.get("A", 0)),
                "音准B(<=35c)个数": int(pitch_counts.get("B", 0)),
                "音准C(<=50c)个数": int(pitch_counts.get("C", 0)),
                "音准D(<=75c)个数": int(pitch_counts.get("D", 0)),
                "音准E(<=100c)个数": int(pitch_counts.get("E", 0)),
                "音准F(成功点中)个数": int(pitch_counts.get("F", 0)),
                "节奏优秀(<=80ms)个数": int(rhythm_counts.get("优秀", 0)),
                "节奏可接受(<=150ms)个数": int(rhythm_counts.get("可接受", 0)),
                "节奏偏差较大(>150ms)个数": int(rhythm_counts.get("偏差较大", 0)),
                "时值A个数": int(dur_counts.get("A", 0)),
                "时值B个数": int(dur_counts.get("B", 0)),
                "时值C个数": int(dur_counts.get("C", 0)),
                "时值D个数": int(dur_counts.get("D", 0)),
                "时值F个数": int(dur_counts.get("F", 0)),
                "低音八度修正个数": octave_fix_count,
                "高八度等效匹配个数": octave_match_count,
                "高八度候选但未放开个数": octave_hold_count,
            }
        ]
    )


# ============================================================
# Excel 着色
# ============================================================
def save_excel_with_highlight(df_align, excel_path):
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_align.to_excel(writer, index=False, sheet_name="对齐结果")
        ws = writer.book["对齐结果"]

        fill_red = PatternFill(fill_type="solid", fgColor="FFC7CE")
        fill_yellow = PatternFill(fill_type="solid", fgColor="FFF2CC")

        headers = [cell.value for cell in ws[1]]
        status_col = headers.index("对齐状态") + 1 if "对齐状态" in headers else None
        susp_col = headers.index("可疑点标记") + 1 if "可疑点标记" in headers else None

        for r in range(2, ws.max_row + 1):
            if status_col is not None:
                v = ws.cell(r, status_col).value
                if v != "对齐成功":
                    for c in range(1, ws.max_column + 1):
                        ws.cell(r, c).fill = fill_red
                    continue

            if susp_col is not None:
                s = ws.cell(r, susp_col).value
                if s in ["可疑点", "连带可疑点"]:
                    for c in range(1, ws.max_column + 1):
                        if ws.cell(r, c).fill.fill_type is None:
                            ws.cell(r, c).fill = fill_yellow


# ============================================================
# 主流程
# ============================================================
def main():
    base_dir = os.path.abspath(
        os.environ.get("ERHU_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
    )

    df_sc, df_on, prefix, score_file, onset_file = load_data(base_dir)

    print(f"✅ score: {score_file}")
    print(f"✅ onset: {onset_file}")
    print(f"🏷️ RUN_TAG = {RUN_TAG}")

    sc_f = df_sc[SCORE_F0_COL].to_numpy(dtype=float)
    on_f = df_on[ONSET_F0_COL].to_numpy(dtype=float)
    sc_m = midi_round(sc_f)
    on_m = midi_round(on_f)

    score_t = get_score_time_array(df_sc)
    onset_t = get_onset_time_array(df_on)

    print("\n[阶段1] 开始主DP匹配...")
    pairs = build_dp_alignment(sc_m, on_m, sc_f, on_f, score_t, onset_t)
    print(f"[阶段1] 主DP完成，得到 {len(pairs)} 对")

    print("\n[阶段1.5] 开始硬前卷 earliest refine...")
    pairs = apply_hard_earliest_refine(
        pairs,
        sc_m=sc_m,
        on_m=on_m,
        sc_f=sc_f,
        on_f=on_f,
        eq_cents=HARD_EARLY_EQ_CENTS,
        lookback_num=LOOKBACK_ONSET_NUM,
    )
    print(f"[阶段1.5] 硬前卷完成，得到 {len(pairs)} 对")

    print("\n[阶段2] 生成结果表...")
    df_align = build_align_table(df_sc, df_on, pairs)
    df_align = append_duration_ratio_columns(df_align)

    if ENABLE_BAD_REGION_DROP:
        print("\n[阶段2.5] 标记可疑点并识别坏区间...")
        df_align, regions = mark_suspicious_and_bad_regions(
            df_align,
            max_gap=BAD_REGION_GAP,
            min_sus_points=BAD_REGION_MIN_SUS,
            extend=BAD_REGION_EXTEND,
        )
        print(f"[阶段2.5] 坏区间数量: {len(regions)}")
        for reg in regions:
            print(
                f"  坏区{reg['region_id']}: "
                f"原可疑 {reg['raw_suspicious_start']}-{reg['raw_suspicious_end']} "
                f"-> 删除 {reg['drop_start']}-{reg['drop_end']} "
                f"(原始可疑点数={reg['suspicious_count']})"
            )
    else:
        regions = []

    print("\n[阶段3] 统计并保存...")
    df_metrics = build_metrics_table(df_align)

    align_csv = os.path.join(base_dir, f"{prefix}_{RUN_TAG}_对齐结果_{METHOD_TAG_CN}.csv")
    metric_csv = os.path.join(base_dir, f"{prefix}_{RUN_TAG}_指标统计_{METHOD_TAG_CN}.csv")
    excel_xlsx = os.path.join(base_dir, f"{prefix}_{RUN_TAG}_对齐结果_{METHOD_TAG_CN}.xlsx")

    df_align.to_csv(align_csv, index=False, encoding="utf-8-sig")
    df_metrics.to_csv(metric_csv, index=False, encoding="utf-8-sig")
    save_excel_with_highlight(df_align, excel_xlsx)

    print(f"✅ 已保存: {align_csv}")
    print(f"✅ 已保存: {metric_csv}")
    print(f"✅ 已保存: {excel_xlsx}")


if __name__ == "__main__":
    main()
