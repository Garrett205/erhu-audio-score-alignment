import os
import glob
import math
import argparse
import tempfile
import pandas as pd
import music21 as m21

from alignment_contract import ONSET_ID_COL, ensure_onset_identity
from tempo_axis_contract import (
    ANCHOR_COL,
    RAW_TIME_COL,
    SCALE_COL,
    SOURCE_COL,
    STRETCHED_TIME_COL,
    stretch_time,
    validate_pitch_mapping,
)


# ------------------------------------------------------------
# os.py
# 作用：
# 1) 从 *_onset_pre_note.csv 读取：
#    - 第一个红点时间：note_frame>=0 的第一行 note_time(s)
#    - black 终点时间：pre_src=="black" 的最后一行 pre_time(s)
#    得到实际演奏时长 dur
# 2) 从 MusicXML 计算 total_beats（总拍数）
# 3) 计算实际平均 bpm：actual_bpm = total_beats / dur * 60
# 4) 从 *_score_final.csv 读取谱面 bpm（BPM列/第7列）
# 5) 缩放 onset 表中的时间列：t' = t0 + (t - t0) * k，其中 k=actual_bpm/score_bpm
# 6) 将原 onset 表改名为 *_old.csv，新表覆盖写回原路径（align 无需改）
# 7) 同时输出两个 summary：
#    - perform_bpm_summary.csv
#    - tempo_scale_summary.csv
# ------------------------------------------------------------


def find_latest(patterns):
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    hits = list(set(hits))
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


# ---------- 从 onset 表读起止时刻 ----------
def read_times_from_onset(onset_csv: str):
    df = pd.read_csv(onset_csv)

    if "note_frame" not in df.columns or "note_time(s)" not in df.columns:
        raise ValueError("onset_csv 缺少 note_frame / note_time(s) 列。")

    red = df[pd.to_numeric(df["note_frame"], errors="coerce").fillna(-1) >= 0]
    if red.empty:
        raise ValueError("onset_csv 没有任何红点（note_frame>=0）。")
    t_start = float(pd.to_numeric(red.iloc[0]["note_time(s)"], errors="coerce"))

    if "pre_src" not in df.columns or "pre_time(s)" not in df.columns:
        raise ValueError("onset_csv 缺少 pre_src / pre_time(s) 列。")

    pre_src = df["pre_src"].astype(str)
    black = df[pre_src == "black"]
    if black.empty:
        raise ValueError("onset_csv 找不到 black 行（pre_src=='black'）。")
    t_end = float(pd.to_numeric(black.iloc[-1]["pre_time(s)"], errors="coerce"))

    if not math.isfinite(t_start) or not math.isfinite(t_end):
        raise ValueError(f"起止时间不是有限数：start={t_start}, end={t_end}")

    dur = t_end - t_start
    if dur <= 0:
        raise ValueError(f"演奏时长<=0（start={t_start}, end={t_end}）。请检查 onset 输出。")

    return df, t_start, t_end, dur


# ---------- MusicXML 总拍数（beats） ----------
def total_beats_from_musicxml(mxml_path: str) -> float:
    """
    计算总拍数（忽略尾部纯休止）：
    - 用全局 offset 找到最后一个 Note/Chord 的结束时间 end_q（quarterLength）
    - beats 只累计到 end_q（end_q 之后全是休止/空小节则不算）
    """
    score = m21.converter.parse(mxml_path)
    part = score.parts[0] if score.parts else score

    events = list(part.recurse().getElementsByClass((m21.note.Note, m21.chord.Chord)))
    if events:
        end_q = 0.0
        for el in events:
            try:
                start = float(el.getOffsetInHierarchy(part))
            except Exception:
                start = float(el.offset)
            dur = float(el.duration.quarterLength)
            end_q = max(end_q, start + dur)
    else:
        end_q = float(score.duration.quarterLength)

    measures = list(part.getElementsByClass(m21.stream.Measure))
    if not measures:
        return float(end_q)

    last_ts = None
    beats_total = 0.0

    for m in measures:
        ts = m.timeSignature or last_ts
        if ts is None:
            ts = m21.meter.TimeSignature("4/4")
        last_ts = ts

        bar_ql = float(m.barDuration.quarterLength)
        beat_ql = float(ts.beatDuration.quarterLength)
        if beat_ql <= 0 or bar_ql <= 0:
            continue

        m_start = float(m.offset)
        m_end = m_start + bar_ql

        if m_start >= end_q:
            break

        overlap = min(m_end, end_q) - m_start
        if overlap > 0:
            beats_total += overlap / beat_ql

    return float(beats_total)


# ---------- 读 score_final 第7列/列名 BPM ----------
def read_score_bpm_from_score_final(score_final_csv: str) -> float:
    df = pd.read_csv(score_final_csv)

    if "BPM" in df.columns:
        s = df["BPM"]
    elif df.shape[1] >= 7:
        s = df.iloc[:, 6]
    else:
        raise ValueError("score_final.csv 没有 BPM 列（也没有第7列）。")

    vals = pd.to_numeric(s, errors="coerce").dropna().values
    for v in vals:
        if v > 0 and math.isfinite(v):
            return float(v)

    raise ValueError("score_final.csv 的 BPM 列里没有可用正数。")


def integrated_score_span(score_frame: pd.DataFrame) -> tuple[float, float, float]:
    required = ["累计时间(s)", "逻辑结束时间(s)", "参与起音对齐"]
    missing = [column for column in required if column not in score_frame.columns]
    if missing:
        raise ValueError(f"score_final 缺少积分秒数字段: {missing}")
    flag = score_frame["参与起音对齐"].fillna("").astype(str).str.strip().str.lower()
    participating = ~flag.isin(["否", "no", "false", "0", ""])
    start_values = pd.to_numeric(
        score_frame.loc[participating, "累计时间(s)"], errors="coerce"
    )
    start_values = start_values[start_values.map(math.isfinite)]
    if start_values.empty:
        raise ValueError("score_final 没有可靠的参与起音对齐逻辑音符")
    end_values = pd.to_numeric(score_frame["逻辑结束时间(s)"], errors="coerce")
    end_values = end_values[end_values.map(math.isfinite)]
    if end_values.empty:
        raise ValueError("score_final 没有可靠的逻辑结束时间")
    score_start = float(start_values.iloc[0])
    score_end = float(end_values.max())
    score_span = score_end - score_start
    if not all(math.isfinite(value) for value in (score_start, score_end, score_span)) or score_span <= 0:
        raise ValueError(f"积分谱面跨度无效: start={score_start}, end={score_end}")
    return score_start, score_end, score_span


def determine_global_scale(
    score_frame: pd.DataFrame,
    performance_start: float,
    performance_end: float,
    *,
    legacy_total_beats: float | None = None,
    legacy_score_bpm: float | None = None,
) -> dict[str, object]:
    performance_span = float(performance_end) - float(performance_start)
    if not math.isfinite(performance_span) or performance_span <= 0:
        raise ValueError(f"演奏跨度无效: {performance_span}")
    fallback = False
    try:
        score_start, score_end, score_span = integrated_score_span(score_frame)
        source = "mxml_integrated_duration"
    except ValueError as exc:
        fallback = True
        if legacy_total_beats is None or legacy_score_bpm is None:
            raise ValueError(f"积分秒数不可用且缺少旧算法回退参数: {exc}") from exc
        if legacy_score_bpm <= 0 or legacy_total_beats <= 0:
            raise ValueError("旧算法回退参数必须为正数")
        score_start = 0.0
        score_span = float(legacy_total_beats) / float(legacy_score_bpm) * 60.0
        score_end = score_start + score_span
        source = "legacy_single_bpm_fallback"
    return {
        "score_start": score_start,
        "score_end": score_end,
        "score_span": score_span,
        "performance_start": float(performance_start),
        "performance_end": float(performance_end),
        "performance_span": performance_span,
        "global_scale": score_span / performance_span,
        "scale_source": source,
        "fallback_to_single_bpm": fallback,
    }


# ---------- 缩放某个时间列（相对锚点 t0） ----------
def scale_time_col(df: pd.DataFrame, col: str, t0: float, k: float) -> None:
    if col not in df.columns:
        return
    x = pd.to_numeric(df[col], errors="coerce")
    mask = x.notna()
    # Keep full precision: pitch2 uses this exact same formula and TRUE
    # verifies it to 1e-6 seconds.
    df.loc[mask, col] = stretch_time(x[mask], t0, k)


def scaled_output_path(onset_csv: str) -> str:
    suffix = "_onset_pre_note.csv"
    if onset_csv.endswith(suffix):
        return onset_csv[: -len(suffix)] + "_onset_scaled.csv"
    root, _ext = os.path.splitext(onset_csv)
    return root + "_scaled.csv"


def build_scaled_onset(
    df_onset: pd.DataFrame,
    t0: float,
    k: float,
    tempo_scale_source: str,
) -> pd.DataFrame:
    """Scale from the immutable raw table; calling this twice with raw input is idempotent."""
    df_new = ensure_onset_identity(df_onset)
    for col in ["note_time(s)", "pre_time(s)", "stable_start_time(s)"]:
        scale_time_col(df_new, col, t0=t0, k=k)
    if "pre_to_note_ms" in df_new.columns:
        values = pd.to_numeric(df_new["pre_to_note_ms"], errors="coerce")
        df_new["pre_to_note_ms"] = (values * k).round(1)
    df_new[SOURCE_COL] = str(tempo_scale_source)
    df_new["tempo_scale_factor"] = float(k)
    df_new[ANCHOR_COL] = float(t0)
    df_new[SCALE_COL] = float(k)
    return df_new


def raw_pitch_output_path(raw_pitch_csv: str) -> str:
    suffix = "_CREPE_plain_raw.csv"
    if raw_pitch_csv.endswith(suffix):
        return raw_pitch_csv[: -len(suffix)] + "_pitch2.csv"
    root, _ext = os.path.splitext(raw_pitch_csv)
    return root + "_pitch2.csv"


def choose_raw_pitch_column(frame: pd.DataFrame) -> str:
    """Use the current raw pitch front-end's canonical onset/pitch field."""
    preferred = [
        "onset频率(Hz)",
        "频段控制频率(Hz)",
        "CREPE中值5(Hz)",
        "频率_raw(Hz)",
    ]
    for column in preferred:
        if column in frame.columns:
            return column
    raise ValueError(f"原始 CREPE pitch 缺少可用音高列，现有列: {frame.columns.tolist()}")


def build_scaled_pitch2(
    raw_pitch: pd.DataFrame,
    t0: float,
    k: float,
    tempo_scale_source: str,
) -> pd.DataFrame:
    """Build pitch2 strictly from immutable CREPE raw time, never old pitch2."""
    if RAW_TIME_COL not in raw_pitch.columns:
        raise ValueError(f"原始 CREPE pitch 缺少时间列: {RAW_TIME_COL}")
    result = raw_pitch.copy()
    raw_time = pd.to_numeric(result[RAW_TIME_COL], errors="coerce")
    if raw_time.notna().sum() == 0:
        raise ValueError("原始 CREPE pitch 没有有效 时间(s)")
    pitch_column = choose_raw_pitch_column(result)
    # Retain these compatibility columns for TRUE/showa while retaining all
    # raw front-end fields for auditability.
    result["time_raw_in_pitch1(s)"] = raw_time
    result["pitch(Hz)"] = pd.to_numeric(result[pitch_column], errors="coerce")
    result[STRETCHED_TIME_COL] = stretch_time(raw_time, t0, k)
    result[ANCHOR_COL] = float(t0)
    result[SCALE_COL] = float(k)
    result[SOURCE_COL] = str(tempo_scale_source)
    result["pitch_source_column"] = pitch_column
    validate_pitch_mapping(result)
    return result


def write_csv_temp(frame: pd.DataFrame, destination: str) -> str:
    """Write a fully materialised sibling temp file before any replacement."""
    directory = os.path.dirname(os.path.abspath(destination))
    fd, temp_path = tempfile.mkstemp(prefix=".tempo_axis_", suffix=".csv", dir=directory)
    os.close(fd)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        return temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def replace_scaled_pair(
    onset_frame: pd.DataFrame,
    onset_path: str,
    pitch_frame: pd.DataFrame,
    pitch_path: str,
) -> None:
    """Validate both outputs before replacing either final file."""
    validate_pitch_mapping(pitch_frame)
    onset_temp = write_csv_temp(onset_frame, onset_path)
    pitch_temp = write_csv_temp(pitch_frame, pitch_path)
    old_onset = None
    old_pitch = None
    if os.path.exists(onset_path):
        with open(onset_path, "rb") as handle:
            old_onset = handle.read()
    if os.path.exists(pitch_path):
        with open(pitch_path, "rb") as handle:
            old_pitch = handle.read()
    onset_replaced = False
    try:
        os.replace(onset_temp, onset_path)
        onset_replaced = True
        os.replace(pitch_temp, pitch_path)
    except Exception:
        # Do not leave a freshly scaled onset paired with an older pitch2.
        if onset_replaced:
            if old_onset is None:
                if os.path.exists(onset_path):
                    os.unlink(onset_path)
            else:
                with open(onset_path, "wb") as handle:
                    handle.write(old_onset)
        if old_pitch is not None and not os.path.exists(pitch_path):
            with open(pitch_path, "wb") as handle:
                handle.write(old_pitch)
        for temp_path in (onset_temp, pitch_temp):
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        raise


# ---------- 写平均演奏 BPM summary（兼容原 bpm.py 输出） ----------
def write_perform_bpm_summary(base_dir: str, onset_csv: str, mxml: str,
                              t_start: float, t_end: float, dur: float,
                              beats: float, avg_bpm: float):
    out_csv = os.path.join(base_dir, "perform_bpm_summary.csv")
    row = {
        "onset_csv": os.path.basename(onset_csv),
        "musicxml": os.path.basename(mxml),
        "start_red_time(s)": round(t_start, 3),
        "black_time(s)": round(t_end, 3),
        "duration(s)": round(dur, 3),
        "total_beats": round(beats, 3),
        "avg_bpm": round(avg_bpm, 3),
    }
    pd.DataFrame([row]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset_csv", type=str, default=None, help="指定 *_onset_pre_note.csv（不填则自动找最新）")
    ap.add_argument("--score_final", type=str, default=None, help="指定 *_score_final.csv（不填则自动找最新）")
    ap.add_argument("--musicxml", type=str, default=None, help="指定 .musicxml（不填则自动找最新）")
    ap.add_argument("--output_csv", type=str, default=None, help="缩放结果路径；默认 *_onset_scaled.csv")
    ap.add_argument("--summary_csv", type=str, default="tempo_scale_summary.csv", help="tempo summary 输出文件名")
    ap.add_argument("--raw_pitch_csv", type=str, default=None, help="raw *_CREPE_plain_raw.csv")
    ap.add_argument("--pitch2_csv", type=str, default=None, help="output *_pitch2.csv")
    args = ap.parse_args()

    base_dir = os.path.abspath(
        os.environ.get("ERHU_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
    )

    onset_csv = args.onset_csv or find_latest([
        os.path.join(base_dir, "*_onset_pre_note.csv"),
        os.path.join(base_dir, "*onset*pre*note*.csv"),
    ])
    if not onset_csv:
        raise FileNotFoundError("没找到 *_onset_pre_note.csv（请先运行 onset.py）。")
    if os.path.basename(onset_csv).endswith("_onset_scaled.csv"):
        raise ValueError("--onset_csv 必须指向原始 *_onset_pre_note.csv，禁止再次缩放 scaled 输出。")

    raw_prefix = os.path.basename(onset_csv)
    if raw_prefix.endswith("_onset_pre_note.csv"):
        raw_prefix = raw_prefix[: -len("_onset_pre_note.csv")]
    default_raw_pitch = os.path.join(base_dir, f"{raw_prefix}_CREPE_plain_raw.csv")
    if args.raw_pitch_csv:
        raw_pitch_csv = os.path.abspath(args.raw_pitch_csv)
    elif os.path.exists(default_raw_pitch):
        raw_pitch_csv = default_raw_pitch
    else:
        raw_pitch_candidates = sorted(glob.glob(os.path.join(base_dir, "*_CREPE_plain_raw.csv")))
        if len(raw_pitch_candidates) != 1:
            raise FileNotFoundError(
                "未找到与原始 onset 同前缀的唯一 *_CREPE_plain_raw.csv；请通过 --raw_pitch_csv 指定。"
            )
        raw_pitch_csv = raw_pitch_candidates[0]
    if not os.path.exists(raw_pitch_csv):
        raise FileNotFoundError(f"未找到原始 CREPE pitch: {raw_pitch_csv}")

    score_final_csv = args.score_final or find_latest([
        os.path.join(base_dir, "*_score_final.csv")
    ])
    if not score_final_csv:
        raise FileNotFoundError("没找到 *_score_final.csv（请先运行 musicxml->score 脚本）。")

    mxml = args.musicxml or find_latest([
        os.path.join(base_dir, "*.musicxml"),
        os.path.join(base_dir, "*.xml"),
    ])
    if not mxml:
        raise FileNotFoundError("没找到 musicxml 文件（*.musicxml / *.xml）。")

    # 1) onset 起止
    df_onset, t0, t_black, dur = read_times_from_onset(onset_csv)
    if "tempo_scale_source" in df_onset.columns:
        raise ValueError("输入 onset 表已经带有缩放元数据，拒绝二次缩放。")

    # 2) 优先使用 mxml.py 已积分的绝对秒数；只有积分字段不可用时，
    #    才读取总拍数和单一 BPM 走旧算法回退。
    score_frame = pd.read_csv(score_final_csv)
    try:
        scale_meta = determine_global_scale(score_frame, t0, t_black)
        beats = math.nan
        score_bpm = math.nan
        actual_bpm = math.nan
    except ValueError:
        beats = total_beats_from_musicxml(mxml)
        score_bpm = read_score_bpm_from_score_final(score_final_csv)
        actual_bpm = beats / dur * 60.0
        scale_meta = determine_global_scale(
            score_frame,
            t0,
            t_black,
            legacy_total_beats=beats,
            legacy_score_bpm=score_bpm,
        )

    # 4) 先写 perform_bpm_summary（兼容原 bpm.py）
    k = float(scale_meta["global_scale"])
    if bool(scale_meta["fallback_to_single_bpm"]):
        print("⚠️ fallback_to_single_bpm = true")

    # 6) 生成缩放后的 onset 表（锚定第一个红点 t0）
    original_csv = onset_csv
    output_csv = os.path.abspath(args.output_csv) if args.output_csv else scaled_output_path(original_csv)
    if os.path.abspath(output_csv) == os.path.abspath(original_csv):
        raise ValueError("缩放输出不得覆盖原始 onset 文件")
    pitch2_csv = os.path.abspath(args.pitch2_csv) if args.pitch2_csv else raw_pitch_output_path(raw_pitch_csv)
    if os.path.abspath(pitch2_csv) == os.path.abspath(raw_pitch_csv):
        raise ValueError("pitch2 输出不得覆盖原始 *_CREPE_plain_raw.csv")
    tempo_scale_source = str(scale_meta["scale_source"])
    df_new = build_scaled_onset(
        df_onset, t0=t0, k=k, tempo_scale_source=tempo_scale_source
    )
    df_pitch2 = build_scaled_pitch2(
        pd.read_csv(raw_pitch_csv), t0=t0, k=k, tempo_scale_source=tempo_scale_source
    )
    raw_identity = ensure_onset_identity(df_onset)[ONSET_ID_COL].astype(str).tolist()
    scaled_identity = df_new[ONSET_ID_COL].astype(str).tolist()
    if raw_identity != scaled_identity:
        raise ValueError("raw/scaled onset_id 集合或顺序发生变化")
    replace_scaled_pair(df_new, output_csv, df_pitch2, pitch2_csv)
    perform_bpm_summary_path = write_perform_bpm_summary(
        base_dir, onset_csv, mxml, t0, t_black, dur, beats, actual_bpm
    )

    # 8) 写 tempo_scale_summary（兼容原 os1.py）
    summary_path = os.path.join(base_dir, args.summary_csv)
    summary = pd.DataFrame([{
        "source_onset_csv": os.path.basename(original_csv),
        "scaled_onset_csv": os.path.basename(output_csv),
        "raw_onset_file": os.path.basename(original_csv),
        "scaled_onset_file": os.path.basename(output_csv),
        "raw_pitch_file": os.path.basename(raw_pitch_csv),
        "pitch2_file": os.path.basename(pitch2_csv),
        "tempo_scale_anchor_s": float(t0),
        "global_scale": float(k),
        "tempo_scale_source": tempo_scale_source,
        "score_final_csv": os.path.basename(score_final_csv),
        "musicxml": os.path.basename(mxml),
        "start_red_time(s)": round(t0, 3),
        "black_time(s)": round(t_black, 3),
        "duration(s)": round(dur, 3),
        "total_beats": round(beats, 3),
        "actual_bpm": round(actual_bpm, 3),
        "score_bpm": round(score_bpm, 3),
        "scale_k(actual/score)": round(k, 6),
        **scale_meta,
    }])
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 9) 打印
    print("==== os (perform bpm + tempo scaling) ====")
    print(f"onset(raw)     : {os.path.basename(original_csv)}  <-- 保持不变")
    print(f"onset(scaled)  : {os.path.basename(output_csv)}  <-- DP 阶段优先读取")
    print(f"score_final    : {os.path.basename(score_final_csv)}  (score_bpm={score_bpm:.3f})")
    print(f"musicxml       : {os.path.basename(mxml)}  (total_beats={beats:.3f})")
    print(f"start_red_time : {t0:.3f}")
    print(f"black_time     : {t_black:.3f}")
    print(f"duration(s)    : {dur:.3f}")
    print(f"actual_bpm     : {actual_bpm:.3f}")
    print(f"global_scale   : {k:.9f}   (t' = t0 + (t-t0)*global_scale)")
    print(f"scale_source   : {scale_meta['scale_source']}")
    print(f"perform_bpm_csv: {os.path.basename(perform_bpm_summary_path)}")
    print(f"summary_csv    : {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
