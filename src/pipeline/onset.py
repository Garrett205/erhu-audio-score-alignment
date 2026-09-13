import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import torch
import torchcrepe

# =========================
# 基础参数（采样步长：5ms）
# =========================
BASE_DIR = os.path.abspath(
    os.environ.get("ERHU_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
)

SR = 16000
HOP = 80                # 5ms @ 16k
DT = HOP / SR

DB_THRESHOLD = 3.0      # “相对强度(dB)”门控阈值

# 能量峰检测参数（蓝点）
# 不再写死 0.07，改为运行时使用 onset_strength_norm 的 P70
ENERGY_PROM = None
ENERGY_PROM_PERCENTILE = 60.0
ENERGY_MIN_IOI_MS = 30

# =========================
# ✅ 黄点（y）评分逻辑：只做候选召回
# 目标：先把“值得保留的结构变化”打出来，不负责 red 主点
# 核心思想：
# 1) 用 pitch 局部数学特征给每一帧一个 y_score
# 2) 高分连续区形成 yellow candidate band
# 3) 每个 band 取一个“偏前”的代表点作为黄点
# 4) 强调宁多勿漏，尤其装饰链、滑音前缘、持续转向
# =========================
# --- 特征窗口 ---
Y_RANGE_HALF_WIN = 2            # R(i): 局部范围窗口半径（共5帧）
Y_CONT_HALF_WIN = 2             # C(i): 持续转向窗口半径

# --- 特征裁剪（用于归一化）---
Y_V_CLIP_CENTS = 120.0          # |v| 裁剪上限（cents / 5ms）
Y_A_CLIP_CENTS = 160.0          # |a| 裁剪上限
Y_R_CLIP_CENTS = 220.0          # R  裁剪上限
Y_C_CLIP_CENTS = 220.0          # C  裁剪上限

# --- 评分权重 ---
Y_W_V = 0.22                    # 跳变强度
Y_W_A = 0.23                    # 转折强度
Y_W_R = 0.20                    # 局部范围
Y_W_C = 0.25                    # 持续转向
Y_W_E = 0.07                    # 能量辅助
Y_W_Z = 0.03                    # 轻惩罚（低peri/弱能量/无效帧）

# --- 候选带提取 ---
Y_SCORE_TH = 0.43               # yellow candidate band 阈值
Y_BAND_MERGE_GAP = 2            # 候选带间允许的小空隙（帧）
Y_MIN_BAND_LEN = 1              # 候选带最短长度（帧）
Y_REP_PEAK_RATIO = 0.92         # band 内取 >= max*ratio 的最早点
Y_MIN_IOI_MS = 20               # 黄点最小间隔：比旧版更宽松，防漏装饰链
Y_MIN_DB_FOR_Y = 0.0            # 极弱能量时仍可留候选，这里只做轻门控
Y_LOW_PERI_TH = 0.01            # 低 periodicity 轻惩罚阈值

# =========================
# ✅ plateau pre：稳定平台进入候选（补逻辑，不替换黄点）
# 用于补充“连续滑动/过渡后进入新稳定音高，但没有明显能量峰或跳变峰”的情况
# 典型例子：滑音落到目标音后形成短稳定平台，yellow 未必有高分，但它应作为候选结构点保留。
# =========================
ENABLE_PLATEAU_PRE = True
PLATEAU_WIN_FRAMES = 6                 # 30ms 稳定窗口
PLATEAU_VALID_RATIO = 0.80             # 窗口内有效音高比例
PLATEAU_RANGE_CENTS_TH = 30.0          # 平台内部音高波动阈值；略放宽以适应二胡滑后短稳定
PLATEAU_MIN_DB = 6.0                   # 平台平均能量下限
PLATEAU_MIN_PERI = 0.05                # 平台平均 periodicity 下限

# 前方参考音高不要取“紧贴平台前的滑音尾巴”，而是先跳过一段，再向前取参考窗口
PLATEAU_REF_GAP_FRAMES = 20            # 跳过当前平台前 100ms，避免把滑音尾巴当作前平台
PLATEAU_REF_WIN_FRAMES = 24            # 再向前取 120ms 作为前方参考音高
PLATEAU_MIN_CHANGE_CENTS = 90.0        # 当前平台与前方参考音高差值阈值，略放宽

PLATEAU_MIN_IOI_MS = 80                # plateau 候选之间最小间隔
PLATEAU_EXISTING_GAP_MS = 20           # 与已有 energy/yellow 候选太近才屏蔽；避免误杀新平台

# =========================
# platform_transition_weak：左右稳定平台转换。
# 它不进入 energy/yellow/plateau 的 strong merge，也不进入 DP1/DP2；
# 仅供 DP3 在“已保守恢复 + 紧邻仍未匹配”的局部缺口中补救。
# =========================
ENABLE_PLATFORM_TRANSITION_WEAK = True
PLATFORM_WEAK_MIN_CONFIDENCE = 0.80
PLATFORM_WEAK_WIN_FRAMES = 6                  # 左右各 30ms
PLATFORM_WEAK_MAX_TRANSITION_FRAMES = 4       # 两平台之间最多允许 20ms 过渡
PLATFORM_WEAK_VALID_RATIO = 0.80
PLATFORM_WEAK_RANGE_CENTS_TH = 35.0
PLATFORM_WEAK_DIFF_CENTS_TH = 70.0
PLATFORM_WEAK_LOCAL_CHANGE_CENTS_TH = 30.0
PLATFORM_WEAK_MIN_DB = 3.0
PLATFORM_WEAK_MIN_PERI = 0.03
PLATFORM_WEAK_STRONG_GAP_MS = 25
# 仅用于调试显示参考，不再作为黄点主逻辑
DELTA_SMOOTH = 3
LAG_FRAMES_LOCAL = 4
PITCH_MIN_IOI_MS = 30

# CREPE 配置（现场运行备份）
CREPE_FMIN = 100
CREPE_FMAX = 2000
CREPE_MODEL = "tiny"
CREPE_BATCH = 16384
CREPE_CHUNK_SEC = 30

# 噪声过滤
CUT_NOISE_HZ = None

# 融合与去重参数
MERGE_TOL_FRAMES = 2
FINAL_MIN_IOI_MS = 30

# =========================
# red（主对齐点）规则
# =========================
PRE_TO_RED_MAX_MS = 80
STABLE_WIN_FRAMES = 6
STABLE_RANGE_CENTS_TH_RED = 30.0
STABLE_VALID_RATIO = 0.80
NOTE_OFFSET_FRAMES = 0
NOTE_F0_WIN_FRAMES = 4

# 主 onset 音高因高频 hard-invalid 被清零时，保守地使用前端已经给出的
# 频段控制轨作为 red 回退。该回退不生成新的 pre，不改变 energy/yellow/
# plateau 门限；只处理已有 pre 后的连续失效片段，并且每个失效片段最多
# 输出一个 red。
ENABLE_CONTROL_RESCUE_RED = True
CONTROL_RESCUE_INVALID_RATIO = 0.80
CONTROL_RESCUE_PRIMARY_ZERO_RATIO = 0.80
CONTROL_RESCUE_RANGE_CENTS_TH = 30.0

# =========================
# Start/End Gate
# =========================
START_GATE_CUT_HZ = 1800.0
START_GATE_PERI_TH = 0.01
START_GATE_OK_FR = 40
START_GATE_BACK_FR = 40

END_GATE_CUT_HZ = 1800.0
END_GATE_PERI_TH = 0.01
END_GATE_OK_FR = 30
END_GATE_BACK_FR = -28


# =========================
# 文件查找与数据加载
# =========================
def pick_first_wav():
    wavs = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".wav")]
    if not wavs:
        raise SystemExit("❌ 当前目录没找到 wav")
    return wavs[0]

def find_pitch_csv(stem: str):
    candidates = [
        os.path.join(BASE_DIR, f"{stem}_CREPE_plain_raw.csv"),
        os.path.join(BASE_DIR, f"{stem}_CREPE_plain_cut1800.csv"),
        os.path.join(BASE_DIR, f"{stem}_精修音高_5ms.csv"),
        os.path.join(BASE_DIR, f"{stem}_pitch_5ms.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for f in os.listdir(BASE_DIR):
        fl = f.lower()
        if fl.endswith(".csv") and (stem in f) and ("crepe" in fl or "pitch" in fl) and ("_old" not in fl):
            return os.path.join(BASE_DIR, f)
    return None

def load_volume_5ms(stem: str):
    volume_csv = os.path.join(BASE_DIR, f"{stem}_相对动态_5ms.csv")
    if not os.path.exists(volume_csv):
        raise SystemExit(f"❌ 找不到音量表：{volume_csv}（请先运行 PMSDB.py）")
    dfv = pd.read_csv(volume_csv)
    vol_col = "相对强度(dB)" if "相对强度(dB)" in dfv.columns else dfv.columns[1]
    vol = dfv[vol_col].values.astype(np.float32)
    return vol, volume_csv

def load_onset_strength_5ms(stem: str):
    onset_csv = os.path.join(BASE_DIR, f"{stem}_onset强度_5ms.csv")
    if not os.path.exists(onset_csv):
        raise SystemExit(f"❌ 找不到 onset 强度表：{onset_csv}（请先运行 PMSDB.py）")

    dfo = pd.read_csv(onset_csv)

    if "onset_strength_norm" in dfo.columns:
        onset_env_n = dfo["onset_strength_norm"].values.astype(np.float32)
    else:
        raise SystemExit(f"❌ onset 强度表缺少列 onset_strength_norm：{onset_csv}")

    if "onset_strength" in dfo.columns:
        onset_env = dfo["onset_strength"].values.astype(np.float32)
    else:
        onset_env = onset_env_n.copy()

    return onset_env, onset_env_n, onset_csv

# =========================
# 工具函数
# =========================
def smooth_mavg(x, k):
    if k <= 1:
        return x.astype(np.float32)
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, w, mode="valid").astype(np.float32)

def hz_to_cents_ratio(f2, f1):
    return 1200.0 * np.log2((f2 + 1e-9) / (f1 + 1e-9))

def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mx = float(np.max(x)) if len(x) else 0.0
    if mx <= 1e-9:
        return np.zeros_like(x)
    return (x / mx).astype(np.float32)

def get_percentile_threshold(x: np.ndarray, q: float = 70.0):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))

def pick_peaks(x, prom, min_ioi_ms):
    min_dist = max(1, int(round((min_ioi_ms / 1000.0) / DT)))
    peaks, _ = find_peaks(x, prominence=prom, distance=min_dist)
    return peaks

def merge_events(fr_energy, fr_pitch, tol_frames=2):
    fr_energy = sorted(set(int(i) for i in fr_energy))
    fr_pitch = sorted(set(int(i) for i in fr_pitch))
    used_pitch = set()
    events = []

    for fe in fr_energy:
        hit = None
        for fp in fr_pitch:
            if fp in used_pitch:
                continue
            if abs(fp - fe) <= tol_frames:
                hit = fp
                break
        if hit is not None:
            used_pitch.add(hit)
            events.append((int(round((fe + hit) / 2)), "both"))
        else:
            events.append((fe, "energy"))

    for fp in fr_pitch:
        if fp not in used_pitch:
            events.append((fp, "pitch"))

    events.sort(key=lambda x: x[0])
    return events

def merge_events_multi(fr_energy, fr_yellow, fr_plateau, tol_frames=2):
    """
    多来源 pre 融合：energy / yellow / plateau。
    若多个来源在 tol_frames 内相邻，则合并为复合来源，并取最早帧，偏向候选前缘召回。
    """
    raw = []
    for fr in fr_energy:
        raw.append((int(fr), "energy"))
    for fr in fr_yellow:
        raw.append((int(fr), "yellow"))
    for fr in fr_plateau:
        raw.append((int(fr), "plateau"))

    raw = sorted(raw, key=lambda x: x[0])
    if not raw:
        return []

    events = []
    used = [False] * len(raw)
    for i, (fr, src) in enumerate(raw):
        if used[i]:
            continue
        group = [(fr, src)]
        used[i] = True
        for j in range(i + 1, len(raw)):
            if used[j]:
                continue
            fr2, src2 = raw[j]
            if fr2 - fr > int(tol_frames):
                break
            if abs(fr2 - fr) <= int(tol_frames):
                group.append((fr2, src2))
                used[j] = True
        frames = [g[0] for g in group]
        srcs = sorted(set(g[1] for g in group))
        out_fr = int(min(frames))
        out_src = "+".join(srcs) if len(srcs) > 1 else srcs[0]
        events.append((out_fr, out_src))
    events.sort(key=lambda x: x[0])
    return events


def hz_to_nearest_note_cents(f_hz: float):
    if not np.isfinite(f_hz) or f_hz <= 0:
        return ("", np.nan)
    midi_float = librosa.hz_to_midi(f_hz)
    midi_round = int(np.round(midi_float))
    note = librosa.midi_to_note(midi_round, octave=True)
    cents = (midi_float - midi_round) * 100.0
    return (note, float(cents))

def fmt_note(note: str, cents: float):
    if note == "" or not np.isfinite(cents):
        return ""
    sign = "+" if cents >= 0 else "-"
    return f"{note} {sign}{abs(cents):.1f}c"

def src_to_color(src: str):
    src = str(src)
    if src == "black":
        return "black"
    if "energy" in src and ("yellow" in src or "plateau" in src):
        return "blue"
    if "energy" in src:
        return "blue"
    if "plateau" in src:
        return "orange"
    if "platform_transition_weak" in src:
        return "purple"
    return "yellow"

def cents_robust_range_in_window(f0_win: np.ndarray):
    ref = np.median(f0_win)
    cents = 1200.0 * np.log2((f0_win + 1e-9) / (ref + 1e-9))
    p10 = np.percentile(cents, 10)
    p90 = np.percentile(cents, 90)
    return float(p90 - p10)

# =========================
# 新黄点逻辑：仅跳变型
# =========================
def hz_to_logcents_track(f0: np.ndarray):
    """把 Hz 轨转成 log-frequency cents 轨；无效帧设为 NaN。"""
    f = np.asarray(f0, dtype=np.float32).copy()
    out = np.full_like(f, np.nan, dtype=np.float32)
    mask = np.isfinite(f) & (f > 0)
    out[mask] = 1200.0 * np.log2(f[mask])
    return out


def robust_local_range(x: np.ndarray, l: int, r: int):
    seg = x[l:r]
    seg = seg[np.isfinite(seg)]
    if seg.size < 2:
        return 0.0
    return float(np.max(seg) - np.min(seg))


def compute_yellow_score(f0: np.ndarray, vol_db: np.ndarray, onset_env_n: np.ndarray, peri: np.ndarray):
    """
    只为 y 生成评分：
    Y = w_v|v| + w_a|a| + w_rR + w_cC + w_eE - w_zZ
    其中：
      v: 一阶变化
      a: 二阶变化
      R: 局部范围
      C: 持续转向（前后净位移）
      E: 能量辅助
      Z: 轻惩罚（无效帧、低 peri、很弱能量）
    """
    n = len(f0)
    p = hz_to_logcents_track(f0)

    v = np.zeros(n, dtype=np.float32)
    if n > 1:
        valid_pair = np.isfinite(p[1:]) & np.isfinite(p[:-1])
        v[1:][valid_pair] = (p[1:][valid_pair] - p[:-1][valid_pair]).astype(np.float32)

    a = np.zeros(n, dtype=np.float32)
    if n > 1:
        a[:-1] = (v[1:] - v[:-1]).astype(np.float32)

    range_win = int(Y_RANGE_HALF_WIN) * 2 + 1
    p_series = pd.Series(p)
    roll = p_series.rolling(window=range_win, center=True, min_periods=2)
    R = (roll.max() - roll.min()).fillna(0.0).to_numpy(dtype=np.float32)

    C = np.zeros(n, dtype=np.float32)
    if n > 0:
        idx = np.arange(n)
        left = np.maximum(0, idx - int(Y_CONT_HALF_WIN))
        right = np.minimum(n - 1, idx + int(Y_CONT_HALF_WIN))
        valid_cont = np.isfinite(p[left]) & np.isfinite(p[right])
        C[valid_cont] = np.abs(p[right][valid_cont] - p[left][valid_cont]).astype(np.float32)

    # 特征归一化
    v_n = np.clip(np.abs(v) / float(Y_V_CLIP_CENTS), 0.0, 1.0)
    a_n = np.clip(np.abs(a) / float(Y_A_CLIP_CENTS), 0.0, 1.0)
    r_n = np.clip(R / float(Y_R_CLIP_CENTS), 0.0, 1.0)
    c_n = np.clip(C / float(Y_C_CLIP_CENTS), 0.0, 1.0)

    # 能量辅助：onset_env_n 为主，vol_db 为辅
    vol_pos = np.clip(vol_db - float(DB_THRESHOLD), 0.0, None)
    e_vol = normalize01(vol_pos)
    e_onset = normalize01(onset_env_n)
    e_n = 0.65 * e_onset + 0.35 * e_vol

    # 轻惩罚：无效帧、低 peri、极弱能量
    z_n = np.zeros(n, dtype=np.float32)
    invalid = ~(np.isfinite(f0) & (f0 > 0))
    z_n[invalid] += 1.0
    z_n[peri < float(Y_LOW_PERI_TH)] += 0.45
    z_n[vol_db < float(Y_MIN_DB_FOR_Y)] += 0.25
    z_n = np.clip(z_n, 0.0, 1.0)

    y_score = (
        float(Y_W_V) * v_n +
        float(Y_W_A) * a_n +
        float(Y_W_R) * r_n +
        float(Y_W_C) * c_n +
        float(Y_W_E) * e_n -
        float(Y_W_Z) * z_n
    ).astype(np.float32)

    # 无效帧最终直接置 0，防止假高分
    y_score[invalid] = 0.0

    feat = {
        "p_logcents": p,
        "v": v,
        "a": a,
        "R": R,
        "C": C,
        "e_n": e_n,
        "z_n": z_n,
        "v_n": v_n,
        "a_n": a_n,
        "r_n": r_n,
        "c_n": c_n,
    }
    return y_score, feat



def score_to_bands(score: np.ndarray, th: float = Y_SCORE_TH, merge_gap: int = Y_BAND_MERGE_GAP,
                   min_len: int = Y_MIN_BAND_LEN):
    idx = np.where(score >= float(th))[0].tolist()
    if not idx:
        return []

    bands = []
    s = idx[0]
    e = idx[0]
    for x in idx[1:]:
        if x - e <= int(merge_gap) + 1:
            e = x
        else:
            if e - s + 1 >= int(min_len):
                bands.append((s, e))
            s = x
            e = x
    if e - s + 1 >= int(min_len):
        bands.append((s, e))
    return bands



def pick_y_from_band(score: np.ndarray, band, ratio: float = Y_REP_PEAK_RATIO):
    s, e = int(band[0]), int(band[1])
    seg = score[s:e + 1]
    if len(seg) == 0:
        return None
    mx = float(np.max(seg))
    cand = np.where(seg >= mx * float(ratio))[0]
    if cand.size == 0:
        return int(s + int(np.argmax(seg)))
    return int(s + cand[0])



def find_yellow_by_score(f0: np.ndarray, vol_db: np.ndarray, onset_env_n: np.ndarray,
                         peri: np.ndarray, start_allow_frame: int):
    """
    新版 y：
    1) 先算每帧 y_score
    2) 高分连续区 -> yellow candidate bands
    3) 每个 band 选一个偏前代表点
    4) 做较宽松最小间隔去重
    """
    n = len(f0)
    y_score, feat = compute_yellow_score(f0, vol_db, onset_env_n, peri)
    y_score[:max(0, int(start_allow_frame))] = 0.0

    bands = score_to_bands(y_score)

    yellow = []
    min_gap = max(1, int(round((float(Y_MIN_IOI_MS) / 1000.0) / DT)))
    last_keep = -10**9

    for band in bands:
        fr = pick_y_from_band(y_score, band)
        if fr is None:
            continue
        if fr - last_keep < min_gap:
            # 若太近，保留分数更高者
            if yellow and y_score[fr] > y_score[yellow[-1]]:
                yellow[-1] = int(fr)
                last_keep = int(fr)
            continue
        yellow.append(int(fr))
        last_keep = int(fr)

    return yellow, bands, y_score, feat

def _valid_f0_values(seg):
    seg = np.asarray(seg, dtype=np.float32)
    return seg[np.isfinite(seg) & (seg > 0)]


def _valid_ratio(seg):
    seg = np.asarray(seg, dtype=np.float32)
    if len(seg) == 0:
        return 0.0
    return float(np.mean(np.isfinite(seg) & (seg > 0)))


def _median_valid_f0(seg):
    valid = _valid_f0_values(seg)
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def _abs_cents_between(f1, f2):
    if not (np.isfinite(f1) and np.isfinite(f2)) or f1 <= 0 or f2 <= 0:
        return np.nan
    return float(abs(1200.0 * np.log2((float(f1) + 1e-9) / (float(f2) + 1e-9))))


def find_plateau_pre(f0: np.ndarray, vol_db: np.ndarray, peri: np.ndarray,
                     start_allow_frame: int, existing_frames=None):
    """
    plateau pre：补充“新稳定平台建立点”。
    不替代 yellow，只补 yellow 不敏感的一类：音高经过连续滑动/过渡后稳定到新平台，
    但局部瞬时跳变、二阶转折、能量峰都不强。
    """
    if existing_frames is None:
        existing_frames = []

    n = len(f0)
    win_frames = int(PLATEAU_WIN_FRAMES)
    ref_gap = int(PLATEAU_REF_GAP_FRAMES)
    ref_win = int(PLATEAU_REF_WIN_FRAMES)
    min_gap = max(1, int(round((float(PLATEAU_MIN_IOI_MS) / 1000.0) / DT)))
    existing_gap = max(1, int(round((float(PLATEAU_EXISTING_GAP_MS) / 1000.0) / DT)))

    existing_frames = sorted(int(x) for x in existing_frames if 0 <= int(x) < n)
    out = []
    last_keep = -10**9

    start = max(0, int(start_allow_frame))
    end_limit = n - win_frames

    for s in range(start, max(start, end_limit)):
        e = s + win_frames
        win = f0[s:e]
        if _valid_ratio(win) < float(PLATEAU_VALID_RATIO):
            continue
        valid = _valid_f0_values(win)
        if valid.size < 2:
            continue
        rr = cents_robust_range_in_window(valid)
        if rr > float(PLATEAU_RANGE_CENTS_TH):
            continue
        if float(np.mean(vol_db[s:e])) < float(PLATEAU_MIN_DB):
            continue
        if float(np.mean(peri[s:e])) < float(PLATEAU_MIN_PERI):
            continue

        cur_f = float(np.median(valid))

        # 关键修正：不要用紧贴当前平台前的帧作参考，因为那里常常已经是滑音尾巴。
        # 先跳过 ref_gap，再向前取 ref_win 作为“前平台/前结构”的参考音高。
        prev_r = max(0, s - ref_gap)
        prev_l = max(0, prev_r - ref_win)
        if prev_r <= prev_l:
            continue
        prev_f = _median_valid_f0(f0[prev_l:prev_r])
        if prev_f <= 0:
            continue
        dc = _abs_cents_between(cur_f, prev_f)
        if not np.isfinite(dc) or dc < float(PLATEAU_MIN_CHANGE_CENTS):
            continue

        # 只屏蔽“几乎重合”的已有候选；不再用 80ms 大间隔误杀 plateau。
        if any(abs(int(s) - fr) < existing_gap for fr in existing_frames):
            continue
        if s - last_keep < min_gap:
            continue

        out.append(int(s))
        last_keep = int(s)

    return out


def find_platform_transition_weak(
    f0: np.ndarray,
    vol_db: np.ndarray,
    peri: np.ndarray,
    start_allow_frame: int,
    strong_frames=None,
):
    """Find strict left-platform -> right-platform transitions for diagnostics.

    These candidates never enter the existing energy/yellow/plateau merge.  Each
    returned frame is the earliest qualified start of the new right platform.
    """
    if strong_frames is None:
        strong_frames = []

    n = min(len(f0), len(vol_db), len(peri))
    win = int(PLATFORM_WEAK_WIN_FRAMES)
    max_transition = int(PLATFORM_WEAK_MAX_TRANSITION_FRAMES)
    strong_gap = max(
        1,
        int(round((float(PLATFORM_WEAK_STRONG_GAP_MS) / 1000.0) / DT)),
    )
    strong = sorted(int(frame) for frame in strong_frames if 0 <= int(frame) < n)

    def platform_stats(left: int, right: int):
        pitch_window = f0[left:right]
        valid_ratio = _valid_ratio(pitch_window)
        valid = _valid_f0_values(pitch_window)
        if valid.size < 2:
            return None
        pitch_range = cents_robust_range_in_window(valid)
        return {
            "valid_ratio": float(valid_ratio),
            "range_cents": float(pitch_range),
            "median_f0": float(np.median(valid)),
            "mean_db": float(np.mean(vol_db[left:right])),
            "mean_peri": float(np.mean(peri[left:right])),
        }

    def max_local_change(left: int, right: int) -> float:
        segment = np.asarray(f0[max(0, left):min(n, right)], dtype=np.float64)
        if len(segment) < 2:
            return 0.0
        a = segment[:-1]
        b = segment[1:]
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if not valid.any():
            return 0.0
        changes = np.abs(1200.0 * np.log2((b[valid] + 1e-9) / (a[valid] + 1e-9)))
        return float(np.max(changes)) if changes.size else 0.0

    qualified = []
    first_right = max(int(start_allow_frame), win)
    last_right = n - win
    for right_start in range(first_right, max(first_right, last_right)):
        if any(abs(right_start - frame) < strong_gap for frame in strong):
            continue

        right_stats = platform_stats(right_start, right_start + win)
        if right_stats is None:
            continue
        if right_stats["valid_ratio"] < float(PLATFORM_WEAK_VALID_RATIO):
            continue
        if right_stats["range_cents"] > float(PLATFORM_WEAK_RANGE_CENTS_TH):
            continue
        if right_stats["mean_db"] < float(PLATFORM_WEAK_MIN_DB):
            continue
        if right_stats["mean_peri"] < float(PLATFORM_WEAK_MIN_PERI):
            continue

        best = None
        for transition_frames in range(max_transition + 1):
            left_end = right_start - transition_frames
            left_start = left_end - win
            if left_start < int(start_allow_frame):
                continue
            left_stats = platform_stats(left_start, left_end)
            if left_stats is None:
                continue
            if left_stats["valid_ratio"] < float(PLATFORM_WEAK_VALID_RATIO):
                continue
            if left_stats["range_cents"] > float(PLATFORM_WEAK_RANGE_CENTS_TH):
                continue
            if left_stats["mean_db"] < float(PLATFORM_WEAK_MIN_DB):
                continue
            if left_stats["mean_peri"] < float(PLATFORM_WEAK_MIN_PERI):
                continue

            diff_cents = _abs_cents_between(
                left_stats["median_f0"], right_stats["median_f0"]
            )
            if not np.isfinite(diff_cents) or diff_cents < float(PLATFORM_WEAK_DIFF_CENTS_TH):
                continue
            local_change = max_local_change(left_end - 1, right_start + 2)
            if local_change < float(PLATFORM_WEAK_LOCAL_CHANGE_CENTS_TH):
                continue

            stability_score = 1.0 - min(
                1.0,
                max(left_stats["range_cents"], right_stats["range_cents"])
                / float(PLATFORM_WEAK_RANGE_CENTS_TH),
            )
            confidence = float(np.mean([
                min(left_stats["valid_ratio"], right_stats["valid_ratio"]),
                stability_score,
                min(1.0, diff_cents / 140.0),
                min(1.0, local_change / 60.0),
                min(1.0, min(left_stats["mean_db"], right_stats["mean_db"]) / 6.0),
                min(1.0, min(left_stats["mean_peri"], right_stats["mean_peri"]) / 0.06),
            ]))
            candidate = {
                "pre_frame": int(right_start),
                "pre_src": "platform_transition_weak",
                "candidate_level": "weak",
                "platform_left_f0": left_stats["median_f0"],
                "platform_right_f0": right_stats["median_f0"],
                "platform_diff_cents": float(diff_cents),
                "platform_local_change_cents": float(local_change),
                "candidate_confidence": confidence,
            }
            if best is None or candidate["candidate_confidence"] > best["candidate_confidence"]:
                best = candidate
        if best is not None:
            qualified.append(best)

    # Consecutive qualified windows describe the same new platform.  Keep the
    # earliest frame so the candidate sits at the right platform entrance.
    selected = []
    last_qualified_frame = None
    for candidate in qualified:
        frame = int(candidate["pre_frame"])
        if last_qualified_frame is None or frame - last_qualified_frame > win:
            selected.append(candidate)
        last_qualified_frame = frame
    return selected


def filter_events_by_source(events, vol_db):
    """分来源门控，避免 pitch/yellow 被 3dB 一刀切，同时 plateau 用自身可信阈值。"""
    kept = []
    n = len(vol_db)
    for fr, src in events:
        fr = int(fr)
        if not (0 <= fr < n):
            continue
        src_str = str(src)
        if "plateau" in src_str:
            if vol_db[fr] >= float(PLATEAU_MIN_DB):
                kept.append((fr, src_str))
        elif "yellow" in src_str:
            if vol_db[fr] >= float(Y_MIN_DB_FOR_Y):
                kept.append((fr, src_str))
        elif "energy" in src_str:
            if vol_db[fr] >= float(DB_THRESHOLD):
                kept.append((fr, src_str))
        else:
            if vol_db[fr] >= float(DB_THRESHOLD):
                kept.append((fr, src_str))
    return kept

# =========================
# red 稳态逻辑（不变）
# =========================
def find_stable_start_after_pre(f0_detect: np.ndarray, vol_db: np.ndarray, pre_fr: int,
                                max_after_frames: int,
                                win_frames: int,
                                valid_ratio_th: float,
                                range_cents_th: float,
                                db_th: float):
    n = len(f0_detect)
    search_end = min(n - win_frames, pre_fr + max_after_frames)
    if search_end < pre_fr:
        return None

    for s in range(pre_fr, search_end + 1):
        vwin = vol_db[s:s + win_frames]
        if float(np.mean(vwin)) < db_th:
            continue

        win = f0_detect[s:s + win_frames]
        valid = win[np.isfinite(win) & (win > 0)]
        if valid.size / float(win_frames) < valid_ratio_th:
            continue

        rr = cents_robust_range_in_window(valid)
        if rr <= range_cents_th:
            return s

    return None

def find_control_rescue_start_after_pre(
    f0_primary: np.ndarray,
    f0_control: np.ndarray,
    hard_invalid: np.ndarray,
    vol_db: np.ndarray,
    pre_fr: int,
    max_after_frames: int,
    win_frames: int,
    valid_ratio_th: float,
    range_cents_th: float,
    db_th: float,
):
    """Find one stable control-band red inside a primary hard-invalid run.

    This is deliberately narrower than normal red search: the normal onset
    pitch must be absent for most of the same window and the pre-existing
    front-end hard-invalid flag must agree.  The returned region start lets
    the caller prevent duplicate rescues from multiple pre events in the same
    corrupted segment.
    """
    if not ENABLE_CONTROL_RESCUE_RED:
        return None

    n = min(len(f0_primary), len(f0_control), len(hard_invalid), len(vol_db))
    search_end = min(n - win_frames, int(pre_fr) + int(max_after_frames))
    if search_end < int(pre_fr):
        return None

    for start in range(int(pre_fr), search_end + 1):
        end = start + int(win_frames)
        if float(np.mean(vol_db[start:end])) < float(db_th):
            continue
        if float(np.mean(hard_invalid[start:end])) < float(CONTROL_RESCUE_INVALID_RATIO):
            continue
        primary = f0_primary[start:end]
        primary_zero_ratio = float(np.mean(~np.isfinite(primary) | (primary <= 0)))
        if primary_zero_ratio < float(CONTROL_RESCUE_PRIMARY_ZERO_RATIO):
            continue
        control = f0_control[start:end]
        valid = control[np.isfinite(control) & (control > 0)]
        if valid.size / float(win_frames) < float(valid_ratio_th):
            continue
        if cents_robust_range_in_window(valid) > float(range_cents_th):
            continue

        region_start = start
        while region_start > 0 and bool(hard_invalid[region_start - 1]):
            region_start -= 1
        return int(start), int(region_start)
    return None


def mean_f0_window(f0: np.ndarray, start_fr: int, win_frames: int):
    n = len(f0)
    end = min(n, start_fr + win_frames)
    seg = f0[start_fr:end]
    seg = seg[np.isfinite(seg) & (seg > 0)]
    if seg.size == 0:
        return 0.0
    return float(np.mean(seg))

# =========================
# Start/End Gate
# =========================
def find_start_allow_frame(f0_raw: np.ndarray, peri: np.ndarray, n: int) -> int:
    ok = (f0_raw > 0) & (f0_raw < float(START_GATE_CUT_HZ)) & (peri > float(START_GATE_PERI_TH))
    run = 0
    found_i = None
    for idx in range(n):
        if ok[idx]:
            run += 1
            if run >= int(START_GATE_OK_FR):
                found_i = idx - int(START_GATE_OK_FR) + 1
                break
        else:
            run = 0

    if found_i is None:
        return 0
    return max(0, int(found_i) - int(START_GATE_BACK_FR))

def find_end_cut_frame(f0_raw: np.ndarray, peri: np.ndarray, n: int) -> int:
    ok = (f0_raw > 0) & (f0_raw < float(END_GATE_CUT_HZ)) & (peri > float(END_GATE_PERI_TH))
    run = 0
    found_i = None
    for idx in range(n - 1, -1, -1):
        if ok[idx]:
            run += 1
            if run >= int(END_GATE_OK_FR):
                found_i = idx
                break
        else:
            run = 0

    if found_i is None:
        return n - 1

    return max(0, int(found_i) - int(END_GATE_BACK_FR))


# =========================
# 主程序逻辑
# =========================
def main():
    wav_name = pick_first_wav()
    stem = os.path.splitext(wav_name)[0]
    wav_path = os.path.join(BASE_DIR, wav_name)

    steps = tqdm(total=9, desc="Pipeline", unit="step")

    # 1) 读音频
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y = y.astype(np.float32)
    steps.update(1)

    # 2) 读音量
    vol_db, volume_csv = load_volume_5ms(stem)
    tqdm.write(f"✅ 音量表: {os.path.basename(volume_csv)}")
    steps.update(1)

    # 3) 读 onset strength 表
    onset_env, onset_env_n, onset_csv = load_onset_strength_5ms(stem)
    tqdm.write(f"✅ onset强度表: {os.path.basename(onset_csv)}")

    # 动态使用当前曲目 onset_strength_norm 的 P70 作为蓝点 prominence
    energy_prom_cur = get_percentile_threshold(onset_env_n, ENERGY_PROM_PERCENTILE)
    tqdm.write(f"✅ ENERGY_PROM 使用当前曲目 onset_strength_norm 的 P{ENERGY_PROM_PERCENTILE:.0f} = {energy_prom_cur:.6f}")

    steps.update(1)

    # 4) 读 pitch csv
    pitch_csv = find_pitch_csv(stem)
    if pitch_csv is None:
        raise SystemExit("❌ 找不到 pitch CSV。请先生成 pitch CSV。")

    dfp = pd.read_csv(pitch_csv)
    if dfp.shape[1] >= 1 and str(dfp.columns[0]).lower().startswith("unnamed"):
        dfp = dfp.drop(columns=[dfp.columns[0]])

    if dfp.shape[1] < 6:
        raise SystemExit(
            f"❌ pitch CSV 列数不足（需要>=6列，当前只有 {dfp.shape[1]} 列）：{os.path.basename(pitch_csv)}"
        )

    # ✅ 全部统一用第6列
    f0_detect_raw = dfp.iloc[:, 5].values.astype(np.float32)
    f0_onset = dfp.iloc[:, 5].values.astype(np.float32)
    if "\u9891\u6bb5\u63a7\u5236\u9891\u7387(Hz)" in dfp.columns:
        f0_control = dfp["\u9891\u6bb5\u63a7\u5236\u9891\u7387(Hz)"].values.astype(np.float32)
    else:
        f0_control = np.zeros_like(f0_onset, dtype=np.float32)
    if "hard_invalid" in dfp.columns:
        hard_invalid = (
            dfp["hard_invalid"].fillna(False).astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes"])
            .to_numpy(dtype=bool)
        )
    else:
        hard_invalid = np.zeros_like(f0_onset, dtype=bool)

    tqdm.write(
        f"✅ pitch CSV: {os.path.basename(pitch_csv)} | 全部统一使用第6列:{dfp.columns[5]}"
    )

    if "periodicity" in dfp.columns:
        peri = dfp["periodicity"].values.astype(np.float32)
    else:
        peri = np.ones_like(f0_detect_raw, dtype=np.float32)
        tqdm.write("⚠️ pitch CSV 不含 periodicity 列：start/end gate 的 peri 条件将等价全通过")

    steps.update(1)

    # 5) 对齐与预处理
    n = min(len(vol_db), len(f0_detect_raw), len(f0_onset), len(f0_control), len(hard_invalid), len(peri), len(onset_env_n), int(np.ceil(len(y) / HOP)))
    vol_db = vol_db[:n]
    f0_detect_raw = f0_detect_raw[:n]
    f0_onset = f0_onset[:n]
    f0_control = f0_control[:n]
    hard_invalid = hard_invalid[:n]
    peri = peri[:n]
    onset_env = onset_env[:n]
    onset_env_n = onset_env_n[:n]
    t = np.arange(n) * DT

    if CUT_NOISE_HZ:
        f0_detect_raw = f0_detect_raw.copy()
        f0_detect_raw[f0_detect_raw > float(CUT_NOISE_HZ)] = 0.0
        f0_onset = f0_onset.copy()
        f0_onset[f0_onset > float(CUT_NOISE_HZ)] = 0.0

    # 6) start-gate
    start_allow_frame = find_start_allow_frame(f0_detect_raw, peri, n)
    print(f"✅ start_allow_frame={start_allow_frame}  (t={t[start_allow_frame]:.3f}s)")
    steps.update(1)

    # 7) 蓝点（能量）
    fr_energy = pick_peaks(onset_env_n, prom=energy_prom_cur, min_ioi_ms=ENERGY_MIN_IOI_MS)
    steps.update(1)

    # 8) 仅调试参考：delta_c
    f0_safe = f0_detect_raw.copy()
    f0_safe[f0_safe <= 0] = np.nan

    delta_c = np.zeros(n, dtype=np.float32)
    if n > LAG_FRAMES_LOCAL:
        delta_part = np.abs(hz_to_cents_ratio(f0_safe[LAG_FRAMES_LOCAL:], f0_safe[:-LAG_FRAMES_LOCAL]))
        delta_part = np.nan_to_num(delta_part, nan=0.0, posinf=0.0, neginf=0.0)
        delta_part = smooth_mavg(delta_part, DELTA_SMOOTH)
        delta_c[LAG_FRAMES_LOCAL:] = delta_part

    # 新黄点：只算 y 候选分数与 candidate bands
    fr_pitch, yellow_bands, y_score_track, y_feat = find_yellow_by_score(
        f0=f0_detect_raw,
        vol_db=vol_db,
        onset_env_n=onset_env_n,
        peri=peri,
        start_allow_frame=start_allow_frame
    )

    # ✅ 补逻辑：plateau pre，抓“滑音/连续变化后进入新稳定平台”的漏检类型
    fr_plateau = []
    if ENABLE_PLATEAU_PRE:
        fr_plateau = find_plateau_pre(
            f0=f0_detect_raw,
            vol_db=vol_db,
            peri=peri,
            start_allow_frame=start_allow_frame,
            # 仍传入已有候选，但 plateau 内部只用很小 existing_gap 去重，避免误杀滑后平台
            existing_frames=list(fr_energy) + list(fr_pitch)
        )
        tqdm.write(f"✅ plateau pre 补充候选数: {len(fr_plateau)}")

    # 9) pre 融合去重：energy / yellow / plateau 多来源融合
    events = merge_events_multi(fr_energy, fr_pitch, fr_plateau, tol_frames=MERGE_TOL_FRAMES)
    kept = filter_events_by_source(events, vol_db)

    final = []
    last_fr = -10**9
    min_dist = max(1, int(round((FINAL_MIN_IOI_MS / 1000.0) / DT)))
    for fr, src in kept:
        if fr - last_fr >= min_dist:
            final.append((fr, src))
            last_fr = fr
    steps.update(1)

    # 基于 pre 生成 red
    max_after_frames = max(1, int(round((PRE_TO_RED_MAX_MS / 1000.0) / DT)))

    # weak 不进入 strong merge。达到严格置信度门槛的行会写入同一 onset
    # 表，以便 os.py 对所有候选使用完全相同的时间轴；DP1/DP2 会显式排除
    # 它们，DP3 仅将其用于局部未匹配桥接。
    platform_weak = []
    if ENABLE_PLATFORM_TRANSITION_WEAK:
        platform_weak = find_platform_transition_weak(
            f0=f0_detect_raw,
            vol_db=vol_db,
            peri=peri,
            start_allow_frame=start_allow_frame,
            strong_frames=[frame for frame, _source in final],
        )
        tqdm.write(f"✅ platform_transition_weak 诊断候选数: {len(platform_weak)}")

    platform_weak_rows = []
    for candidate in platform_weak:
        pre_fr = int(candidate["pre_frame"])
        stable_start = find_stable_start_after_pre(
            f0_detect=f0_onset,
            vol_db=vol_db,
            pre_fr=pre_fr,
            max_after_frames=max_after_frames,
            win_frames=int(STABLE_WIN_FRAMES),
            valid_ratio_th=float(STABLE_VALID_RATIO),
            range_cents_th=float(STABLE_RANGE_CENTS_TH_RED),
            db_th=float(DB_THRESHOLD),
        )
        if stable_start is None:
            note_fr = -1
            note_time = np.nan
            note_f0 = 0.0
        else:
            note_fr = int(min(n - 1, stable_start + int(NOTE_OFFSET_FRAMES)))
            note_time = float(t[note_fr])
            note_f0 = mean_f0_window(f0_onset, note_fr, int(NOTE_F0_WIN_FRAMES))
        platform_weak_rows.append({
            "note_time(s)": round(float(note_time), 3) if np.isfinite(note_time) else np.nan,
            "note_f0(Hz)": round(float(note_f0), 2) if note_f0 > 0 else 0.0,
            "note_frame": int(note_fr),
            "pre_time(s)": round(float(t[pre_fr]), 3),
            "pre_frame": pre_fr,
            "pre_src": "platform_transition_weak",
            "candidate_level": "weak",
            "platform_left_f0": round(float(candidate["platform_left_f0"]), 2),
            "platform_right_f0": round(float(candidate["platform_right_f0"]), 2),
            "platform_diff_cents": round(float(candidate["platform_diff_cents"]), 2),
            "platform_local_change_cents": round(float(candidate["platform_local_change_cents"]), 2),
            "candidate_confidence": round(float(candidate["candidate_confidence"]), 6),
        })

    rows = []
    red_end_cut_frame = find_end_cut_frame(f0_detect_raw, peri, n)
    used_control_rescue_regions = set()
    for pre_fr, src in final:
        pre_fr = int(pre_fr)
        pre_time = float(t[pre_fr])

        if pre_fr < start_allow_frame:
            continue

        stable_start = find_stable_start_after_pre(
            f0_detect=f0_onset,
            vol_db=vol_db,
            pre_fr=pre_fr,
            max_after_frames=max_after_frames,
            win_frames=int(STABLE_WIN_FRAMES),
            valid_ratio_th=float(STABLE_VALID_RATIO),
            range_cents_th=float(STABLE_RANGE_CENTS_TH_RED),
            db_th=float(DB_THRESHOLD)
        )
        red_pitch_track = f0_onset
        red_pitch_source = "primary"
        if stable_start is None and pre_fr <= red_end_cut_frame:
            rescue = find_control_rescue_start_after_pre(
                f0_primary=f0_onset,
                f0_control=f0_control,
                hard_invalid=hard_invalid,
                vol_db=vol_db,
                pre_fr=pre_fr,
                max_after_frames=max_after_frames,
                win_frames=int(STABLE_WIN_FRAMES),
                valid_ratio_th=float(STABLE_VALID_RATIO),
                range_cents_th=float(CONTROL_RESCUE_RANGE_CENTS_TH),
                db_th=float(DB_THRESHOLD),
            )
            if rescue is not None:
                rescue_start, rescue_region = rescue
                if rescue_region not in used_control_rescue_regions:
                    used_control_rescue_regions.add(rescue_region)
                    stable_start = rescue_start
                    red_pitch_track = f0_control
                    red_pitch_source = "control_rescue"

        if stable_start is None:
            note_fr = -1
            note_time = np.nan
            note_f0 = 0.0
            note_note = ""
            stable_time = np.nan
        else:
            note_fr = int(min(n - 1, stable_start + int(NOTE_OFFSET_FRAMES)))
            note_time = float(t[note_fr])

            note_f0 = mean_f0_window(red_pitch_track, note_fr, int(NOTE_F0_WIN_FRAMES))
            nn, cc = hz_to_nearest_note_cents(note_f0) if note_f0 > 0 else ("", np.nan)
            note_note = fmt_note(nn, cc)
            stable_time = float(t[int(stable_start)])

            if (note_fr - pre_fr) > max_after_frames:
                note_fr = -1
                note_time = np.nan
                note_f0 = 0.0
                note_note = ""
                stable_time = np.nan

        rows.append({
            "note_time(s)": (round(float(note_time), 3) if np.isfinite(note_time) else np.nan),
            "note_f0(Hz)": (round(float(note_f0), 2) if note_f0 > 0 else 0.0),
            "note_note(cent)": note_note,
            "note_frame": int(note_fr),
            "note_color": "red",
            "red_pitch_source": red_pitch_source,

            "pre_time(s)": round(pre_time, 3),
            "pre_frame": int(pre_fr),
            "pre_src": src,
            "pre_color": src_to_color(src),
            "candidate_level": "strong",
            "platform_left_f0": np.nan,
            "platform_right_f0": np.nan,
            "platform_diff_cents": np.nan,
            "platform_local_change_cents": np.nan,
            "candidate_confidence": np.nan,

            "stable_start_time(s)": (round(float(stable_time), 3) if np.isfinite(stable_time) else np.nan),
            "pre_to_note_ms": (round((note_fr - pre_fr) * DT * 1000.0, 1) if note_fr >= 0 else np.nan),
            "onset_env_norm(pre)": round(float(onset_env_n[pre_fr]), 4),
            "delta_cents(pre)": round(float(delta_c[pre_fr]), 2),
        })

    steps.close()

    df_out = pd.DataFrame(rows)

    # Keep normal rows in their historical order.  Appending weak rows after
    # them preserves every strong onset_id, while the final black row remains
    # the end marker.  A weak candidate on an already-used physical red frame
    # cannot add information and is discarded here.
    strong_note_frames = {
        int(value)
        for value in pd.to_numeric(df_out.get("note_frame", pd.Series(dtype=float)), errors="coerce").dropna()
        if int(value) >= 0
    }
    weak_rows_for_pipeline = []
    for weak_row in platform_weak_rows:
        note_frame = int(weak_row["note_frame"])
        confidence = float(weak_row["candidate_confidence"])
        if (
            note_frame < 0
            or note_frame in strong_note_frames
            or confidence < float(PLATFORM_WEAK_MIN_CONFIDENCE)
        ):
            continue
        weak_rows_for_pipeline.append({
            "note_time(s)": weak_row["note_time(s)"],
            "note_f0(Hz)": weak_row["note_f0(Hz)"],
            "note_note(cent)": "",
            "note_frame": note_frame,
            "note_color": "red",
            "red_pitch_source": "primary",
            "pre_time(s)": weak_row["pre_time(s)"],
            "pre_frame": int(weak_row["pre_frame"]),
            "pre_src": "platform_transition_weak",
            "pre_color": "purple",
            "candidate_level": "weak",
            "platform_left_f0": weak_row["platform_left_f0"],
            "platform_right_f0": weak_row["platform_right_f0"],
            "platform_diff_cents": weak_row["platform_diff_cents"],
            "platform_local_change_cents": weak_row["platform_local_change_cents"],
            "candidate_confidence": weak_row["candidate_confidence"],
            "stable_start_time(s)": weak_row["pre_time(s)"],
            "pre_to_note_ms": round((note_frame - int(weak_row["pre_frame"])) * DT * 1000.0, 1),
            "onset_env_norm(pre)": round(float(onset_env_n[int(weak_row["pre_frame"])]), 4),
            "delta_cents(pre)": round(float(delta_c[int(weak_row["pre_frame"])]), 2),
        })
    if weak_rows_for_pipeline:
        df_out = pd.concat([df_out, pd.DataFrame(weak_rows_for_pipeline)], ignore_index=True)
        tqdm.write(
            f"✅ platform_transition_weak 可供 DP3 局部补救: {len(weak_rows_for_pipeline)}"
        )

    # 终点 black
    end_cut_frame = find_end_cut_frame(f0_detect_raw, peri, n)
    end_cut_time = float(t[end_cut_frame])
    print(f"✅ end_cut_frame={end_cut_frame}  (t={end_cut_time:.3f}s)")

    if len(df_out) == 0 or not (df_out["note_frame"] >= 0).any():
        print("⚠️ 没有任何有效红点（note_frame>=0），无法裁尾/black。仍然输出原表。")
    else:
        first_red_idx = int(df_out.index[(df_out["note_frame"] >= 0)].to_list()[0])
        first_red_time = float(df_out.loc[first_red_idx, "note_time(s)"])
        print(f"✅ start(第一个红点) time={first_red_time:.3f}s")

        keep_mask = (df_out["note_time(s)"].isna()) | (df_out["note_time(s)"] <= end_cut_time + 1e-9)
        df_out = df_out.loc[keep_mask].copy()

        black_row = {
            "note_time(s)": np.nan,
            "note_f0(Hz)": 0.0,
            "note_note(cent)": "",
            "note_frame": -1,
            "note_color": "",
            "red_pitch_source": "",

            "pre_time(s)": round(end_cut_time, 3),
            "pre_frame": int(end_cut_frame),
            "pre_src": "black",
            "pre_color": "black",
            "candidate_level": "",
            "platform_left_f0": np.nan,
            "platform_right_f0": np.nan,
            "platform_diff_cents": np.nan,
            "platform_local_change_cents": np.nan,
            "candidate_confidence": np.nan,

            "stable_start_time(s)": np.nan,
            "pre_to_note_ms": np.nan,
            "onset_env_norm(pre)": np.nan,
            "delta_cents(pre)": np.nan,
        }
        df_out = pd.concat([df_out, pd.DataFrame([black_row])], ignore_index=True)
        print(f"✅ black 终点事件已追加为最后一行：t={end_cut_time:.3f}s")

    # 保存 CSV
    out_csv = os.path.join(BASE_DIR, f"{stem}_onset_pre_note.csv")
    cols = [
        "note_time(s)", "note_f0(Hz)", "note_note(cent)", "note_frame", "note_color", "red_pitch_source",
        "pre_time(s)", "pre_frame", "pre_src", "pre_color", "candidate_level",
        "platform_left_f0", "platform_right_f0", "platform_diff_cents",
        "platform_local_change_cents", "candidate_confidence",
        "stable_start_time(s)", "pre_to_note_ms", "onset_env_norm(pre)", "delta_cents(pre)",
    ]
    cols = [c for c in cols if c in df_out.columns]
    df_out = df_out[cols]
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    pre_count = len(df_out)
    red_valid = int((df_out["note_frame"] >= 0).sum()) if pre_count > 0 else 0
    print(f"✅ 表已保存: {out_csv} | pre_count={pre_count} | red_valid={red_valid}")

    if ENABLE_PLATFORM_TRANSITION_WEAK:
        weak_csv = os.path.join(BASE_DIR, f"{stem}_platform_transition_weak.csv")
        pd.DataFrame(platform_weak_rows, columns=[
            "note_time(s)", "note_f0(Hz)", "note_frame",
            "pre_time(s)", "pre_frame", "pre_src", "candidate_level",
            "platform_left_f0", "platform_right_f0", "platform_diff_cents",
            "platform_local_change_cents", "candidate_confidence",
        ]).to_csv(weak_csv, index=False, encoding="utf-8-sig")
        weak_valid = sum(int(row["note_frame"]) >= 0 for row in platform_weak_rows)
        print(
            f"✅ weak诊断表已保存: {weak_csv} | "
            f"weak_count={len(platform_weak_rows)} | weak_valid_red={weak_valid}"
        )

    # =========================
    # 总图调试
    # =========================
    out_png = os.path.join(BASE_DIR, f"{stem}_onset_pre_note_debug.png")

    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, onset_env_n, linewidth=0.9, label="onset_strength_norm")
    ax1.set_ylabel("onset_env")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, f0_onset, linewidth=0.8, label="pitch(col6)")
    ax2.set_ylabel("Hz")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, vol_db, linewidth=0.8, label="relative dB")
    ax3.axhline(DB_THRESHOLD, color="r", linestyle="--", alpha=0.7, label="db threshold")
    ax3.set_ylabel("dB")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper right")

    if start_allow_frame > 0:
        for ax in (ax1, ax2, ax3):
            ax.axvline(float(t[start_allow_frame]), color="k", linestyle="--", alpha=0.5)
        ax1.text(float(t[start_allow_frame]), 0.95, "start_allow", transform=ax1.get_xaxis_transform(),
                 fontsize=9, va="top", ha="left")

    for ax in (ax1, ax2, ax3):
        ax.axvline(float(end_cut_time), color="k", linestyle="--", alpha=0.8)
    ax1.text(float(end_cut_time), 0.95, "end_cut", transform=ax1.get_xaxis_transform(),
             fontsize=9, va="top", ha="left")

    # 画 yellow candidate bands 参考线
    for s, e in yellow_bands:
        ax2.axvspan(float(t[s]), float(t[min(len(t)-1, e)]), color="gold", alpha=0.14)

    for _, r in df_out.iterrows():
        pre_src = str(r.get("pre_src", ""))
        pre_fr = int(r.get("pre_frame", -1)) if np.isfinite(r.get("pre_frame", np.nan)) else -1
        pre_t = r.get("pre_time(s)", np.nan)

        if pre_src == "black" and np.isfinite(pre_t):
            y_black = float(f0_onset[end_cut_frame]) if 0 <= end_cut_frame < len(f0_onset) else 0.0
            ax2.scatter([float(pre_t)], [y_black], s=90, c="black", edgecolors="white", zorder=12)
            continue

        if pre_fr >= 0 and np.isfinite(pre_t):
            pre_c = str(r.get("pre_color", "blue"))
            ax1.scatter([float(pre_t)], [float(onset_env_n[pre_fr])], s=22, c=pre_c, edgecolors="k", zorder=5)
            ax2.scatter([float(pre_t)], [float(f0_onset[pre_fr])], s=22, c=pre_c, edgecolors="k", zorder=5)
            ax3.scatter([float(pre_t)], [float(vol_db[pre_fr])], s=22, c=pre_c, edgecolors="k", zorder=5)

        note_fr = int(r.get("note_frame", -1))
        note_t = r.get("note_time(s)", np.nan)
        if note_fr >= 0 and np.isfinite(note_t):
            note_f = float(r.get("note_f0(Hz)", 0.0))
            ax1.scatter([float(note_t)], [float(onset_env_n[note_fr])], s=30, c="red", edgecolors="k", zorder=7)
            ax3.scatter([float(note_t)], [float(vol_db[note_fr])], s=30, c="red", edgecolors="k", zorder=7)
            ax2.scatter([float(note_t)], [note_f], s=12, c="red", edgecolors="k", zorder=8)

    plt.suptitle(
        f"Pre (blue=energy, yellow=y_score, orange=plateau) + Note(red, col6)\n"
        f"ENERGY_PROM = current onset_strength_norm P{ENERGY_PROM_PERCENTILE:.0f} | yellow=y_score bands | plateau=stable-platform补点"
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🖼️ Debug 图已保存: {out_png}")

    # Pre-only 图
    out_png_pre = os.path.join(BASE_DIR, f"{stem}_pre_only_debug.png")

    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, onset_env_n, linewidth=0.9, label="onset_strength_norm")
    ax1.set_ylabel("onset_env")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, f0_onset, linewidth=0.8, label="pitch(col6)")
    ax2.set_ylabel("Hz")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, vol_db, linewidth=0.8, label="relative dB")
    ax3.axhline(DB_THRESHOLD, color="r", linestyle="--", alpha=0.7, label="db threshold")
    ax3.set_ylabel("dB")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper right")

    if start_allow_frame > 0:
        for ax in (ax1, ax2, ax3):
            ax.axvline(float(t[start_allow_frame]), color="k", linestyle="--", alpha=0.5)

    for ax in (ax1, ax2, ax3):
        ax.axvline(float(end_cut_time), color="k", linestyle="--", alpha=0.8)

    for s, e in yellow_bands:
        ax2.axvspan(float(t[s]), float(t[min(len(t)-1, e)]), color="gold", alpha=0.14)

    for _, r in df_out.iterrows():
        pre_src = str(r.get("pre_src", ""))
        pre_fr = int(r.get("pre_frame", -1)) if np.isfinite(r.get("pre_frame", np.nan)) else -1
        pre_t = r.get("pre_time(s)", np.nan)

        if pre_src == "black" and np.isfinite(pre_t):
            y_black = float(f0_onset[end_cut_frame]) if 0 <= end_cut_frame < len(f0_onset) else 0.0
            ax2.scatter([float(pre_t)], [y_black], s=90, c="black", edgecolors="white", zorder=12)
            continue

        if pre_fr >= 0 and np.isfinite(pre_t):
            pre_c = str(r.get("pre_color", "blue"))
            ax1.scatter([float(pre_t)], [float(onset_env_n[pre_fr])], s=28, c=pre_c, edgecolors="k", zorder=6)
            ax2.scatter([float(pre_t)], [float(f0_onset[pre_fr])], s=28, c=pre_c, edgecolors="k", zorder=6)
            ax3.scatter([float(pre_t)], [float(vol_db[pre_fr])], s=28, c=pre_c, edgecolors="k", zorder=6)

    plt.suptitle(
        f"Pre Only Debug (blue=energy, yellow=y_score, orange=plateau, black=end) | col6 pitch | ENERGY_PROM=P{ENERGY_PROM_PERCENTILE:.0f}"
    )
    plt.tight_layout()
    plt.savefig(out_png_pre, dpi=150)
    print(f"🖼️ Pre-only 图已保存: {out_png_pre}")


if __name__ == "__main__":
    main()
