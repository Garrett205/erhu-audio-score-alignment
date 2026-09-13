# -*- coding: utf-8 -*-
"""
pitch.py

统一音高前端：一次完成 CREPE、dSWIPE 与最终融合。

正式流程：
    PMSDB.py -> pitch.py -> mxml.py -> onset.py -> ...

输出：
    <stem>_CREPE_plain_raw.csv

前六列固定为：
    1. 时间(s)
    2. 频率_raw(Hz)          # CREPE 原始频率
    3. periodicity
    4. 参考强度(dB)
    5. dSWIPE_raw(Hz)
    6. onset频率(Hz)         # 下游 onset.py 使用的最终音高

设计原则：
- CREPE 先做 5 帧居中中值，仅降低模型抖动；原始 CREPE 列仍完整保留。
- dSWIPE 先做 7 帧“保零、分段”中值，只压制孤立毛刺，不跨越无效区回填。
- periodicity < 0.01 直接清零，不进行低置信回填。
- 0.01 <= periodicity < 0.03 时禁用 CREPE，使用 dSWIPE 中值轨迹。
- periodicity >= 0.03 时按固定频段规则融合；460--520 Hz 与
  1750--1850 Hz 在 log-frequency 空间平滑交接。
- 不做长零回填、不做 250/280 Hz 硬清零、不做最终轨迹的全局平滑。
- 先处理短跳音伪连接：相邻变化至少50 cents，连续陡边中至少两步达到
  局部左右跨度的25%，中间最多15帧；中间按帧数劈半，左半贴紧邻左值，
  右半贴紧邻右值。不找远处平台，不使用局部强度、方向反转或额外形状判断。
- 再处理 1--15 帧、边界跳变位于 1100--1300 cents 的短高八度小岛；
  修正时整段除以2，保留原来的滑音和细微走势。超过15帧一律不修改。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# =========================
# 路径与统一时间轴
# =========================
BASE_DIR = Path(
    os.environ.get("ERHU_WORK_DIR", str(Path(__file__).resolve().parent))
).resolve()
SR = 16000
HOP = 80
HOP_SECONDS = HOP / SR  # 5 ms
ALIGN_TOLERANCE_SECONDS = 0.0075

# =========================
# CREPE
# =========================
CREPE_FMIN = 100
CREPE_FMAX = 2000
CREPE_MODEL = "tiny"
CREPE_BATCH_SIZE = 8192

# =========================
# dSWIPE
# =========================
DSWIPE_FS = 16000
DSWIPE_HOP_SIZE = 80
DSWIPE_F0_MIN = 150
DSWIPE_F0_MAX = 2600

# =========================
# 已确认的融合参数
# =========================
PERIODICITY_HARD_ZERO = 0.01
PERIODICITY_CREPE_MIN = 0.03
DB_HARD_ZERO = 5.0

DSWIPE_MEDIAN_WINDOW = 7

LOW_DSWIPE_END_HZ = 460.0
LOW_CREPE_START_HZ = 520.0
HIGH_CREPE_END_HZ = 1750.0
HIGH_DSWIPE_START_HZ = 1850.0
HARD_MAX_F0_HZ = 2400.0

# 只用于诊断，不修改音高。
JUMP_FLAG_CENTS = 70.0

# 简化短跳音伪连接修正。5 ms/帧，15帧约75 ms。
JUMP_BRIDGE_ALPHA = 0.25
JUMP_BRIDGE_MIN_STEP_CENTS = 50.0
JUMP_BRIDGE_MAX_FRAMES = 15

# 保守短高八度小岛修正。5 ms/帧，15帧约75 ms。
OCTAVE_REPAIR_MAX_FRAMES = 15
OCTAVE_REPAIR_MIN_CENTS = 1100.0
OCTAVE_REPAIR_MAX_CENTS = 1300.0
OCTAVE_REPAIR_CONNECTION_CENTS = 250.0

# 下游 onset.py 固定读取第六列，因此列顺序不可随意调整。
TIME_COL = "时间(s)"
CREPE_COL = "频率_raw(Hz)"
PERIODICITY_COL = "periodicity"
DB_COL = "参考强度(dB)"
DSWIPE_COL = "dSWIPE_raw(Hz)"
FINAL_COL = "onset频率(Hz)"


# =========================
# 文件与外部程序
# =========================
def pick_audio(base_dir: Path = BASE_DIR) -> Tuple[Path, str]:
    """与 PMSDB.py 保持一致：使用目录中 os.listdir 返回的第一个 WAV。"""
    wav_names = [name for name in os.listdir(base_dir) if name.lower().endswith(".wav")]
    if not wav_names:
        raise SystemExit("❌ 当前目录未找到 .wav 文件")
    if len(wav_names) > 1:
        print(f"⚠️ 当前目录发现 {len(wav_names)} 个 WAV；将与 PMSDB.py 一样使用第一个：{wav_names[0]}")
    wav_path = base_dir / wav_names[0]
    return wav_path, wav_path.stem


def load_reference_db(stem: str, base_dir: Path = BASE_DIR) -> np.ndarray:
    path = base_dir / f"{stem}_相对动态_5ms.csv"
    if not path.exists():
        raise SystemExit(f"❌ 找不到音量数据表：{path.name}；请先运行 PMSDB.py")

    df = pd.read_csv(path)
    if "相对强度(dB)" in df.columns:
        col = "相对强度(dB)"
    elif len(df.columns) >= 2:
        col = df.columns[1]
    else:
        raise SystemExit(f"❌ 音量表列数异常：{path.name}")

    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def run_crepe(wav_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """运行未做时间平滑的 CREPE，返回 frequency 与 periodicity。"""
    try:
        import librosa
        import torch
        import torchcrepe
    except ImportError as exc:
        raise SystemExit(
            "❌ 缺少 CREPE 依赖。请安装：pip install librosa torch torchcrepe"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"🚀 CREPE | device={device} | wav={wav_path.name}")
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y = y / peak

    audio = torch.from_numpy(y).unsqueeze(0).to(device)
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            audio,
            SR,
            hop_length=HOP,
            fmin=CREPE_FMIN,
            fmax=CREPE_FMAX,
            model=CREPE_MODEL,
            batch_size=CREPE_BATCH_SIZE,
            device=device,
            return_periodicity=True,
        )

    crepe = pitch.squeeze(0).detach().cpu().numpy().astype(np.float64)
    peri = periodicity.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return crepe, peri


def _read_dswipe_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(c).lower(): c for c in df.columns}

    time_col = lower.get("time")
    freq_col = lower.get("frequency")
    if time_col is None or freq_col is None:
        if len(df.columns) < 2:
            raise RuntimeError(f"dSWIPE 输出列异常：{df.columns.tolist()}")
        time_col, freq_col = df.columns[:2]

    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_col], errors="coerce"),
            "frequency": pd.to_numeric(df[freq_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def run_dswipe(wav_path: Path) -> pd.DataFrame:
    """在临时目录运行 dSWIPE，避免覆盖项目根目录中的同名 CSV。"""
    executable = shutil.which("dswipe")
    if executable is None:
        raise SystemExit("❌ 找不到 dswipe 命令；请安装 df0-pitch 并确认 dswipe 在 PATH 中")

    with tempfile.TemporaryDirectory(prefix="erhu_dswipe_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        cmd = [
            executable,
            str(wav_path),
            "--dir_out",
            str(temp_dir),
            "--fs",
            str(DSWIPE_FS),
            "--hop_size",
            str(DSWIPE_HOP_SIZE),
            "--f0_min",
            str(DSWIPE_F0_MIN),
            "--f0_max",
            str(DSWIPE_F0_MAX),
        ]
        print("🚀 dSWIPE |", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.returncode != 0:
            detail = proc.stderr.strip() or "未知错误"
            raise SystemExit(f"❌ dSWIPE 运行失败：{detail}")

        exact = temp_dir / f"{wav_path.stem}.csv"
        if exact.exists():
            output = exact
        else:
            candidates = sorted(temp_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise SystemExit("❌ dSWIPE 未生成 CSV 输出")
            output = candidates[0]

        return _read_dswipe_csv(output)


def align_dswipe(times: np.ndarray, dswipe_df: pd.DataFrame) -> np.ndarray:
    left = pd.DataFrame({TIME_COL: np.asarray(times, dtype=float)})
    right = dswipe_df.rename(columns={"time": "ds_time", "frequency": DSWIPE_COL})
    merged = pd.merge_asof(
        left.sort_values(TIME_COL),
        right.sort_values("ds_time"),
        left_on=TIME_COL,
        right_on="ds_time",
        direction="nearest",
        tolerance=ALIGN_TOLERANCE_SECONDS,
    )
    return (
        pd.to_numeric(merged[DSWIPE_COL], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


# =========================
# 融合核心（可独立单元测试）
# =========================
def smoothstep01(u: np.ndarray | float) -> np.ndarray:
    u_arr = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return u_arr * u_arr * (3.0 - 2.0 * u_arr)


def log_blend(a_hz: np.ndarray, b_hz: np.ndarray, weight_b: np.ndarray) -> np.ndarray:
    """在 log-frequency 空间从 a 平滑过渡到 b。输入必须为正数。"""
    a = np.asarray(a_hz, dtype=float)
    b = np.asarray(b_hz, dtype=float)
    w = np.asarray(weight_b, dtype=float)
    return np.exp((1.0 - w) * np.log(a) + w * np.log(b))


def positive_median3(values: np.ndarray) -> np.ndarray:
    """仅用于频段权重控制；不会覆盖或平滑最终输出。"""
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x)
    for i in range(len(x)):
        segment = x[max(0, i - 1): min(len(x), i + 2)]
        valid = segment[np.isfinite(segment) & (segment > 0)]
        if valid.size:
            out[i] = float(np.median(valid))
    return out


def centered_median5(values: np.ndarray) -> np.ndarray:
    """CREPE 的轻度前置处理；不修改原始列，也不负责零区回填。"""
    x = np.asarray(values, dtype=float)
    return (
        pd.Series(x)
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )




def segmented_dswipe_median(
    values: np.ndarray,
    db: np.ndarray,
    window: int = DSWIPE_MEDIAN_WINDOW,
) -> np.ndarray:
    """
    对 dSWIPE 做保零、分段中值。

    只在连续有效区间内部做居中中值；原本无效、超上限或 dB<5 的帧保持为0，
    因而不会跨越静音/缺失区进行回填。这里不设置 250/280 Hz 低频硬阈值。
    """
    x = np.asarray(values, dtype=float)
    db_arr = np.asarray(db, dtype=float)
    n = min(len(x), len(db_arr))
    x = x[:n]
    db_arr = db_arr[:n]

    valid = (
        np.isfinite(x)
        & (x > 0)
        & (x <= HARD_MAX_F0_HZ)
        & np.isfinite(db_arr)
        & (db_arr >= DB_HARD_ZERO)
    )
    base = np.where(valid, x, 0.0)
    out = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        if base[i] <= 0:
            i += 1
            continue
        j = i + 1
        while j < n and base[j] > 0:
            j += 1
        out[i:j] = (
            pd.Series(base[i:j])
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
        i = j

    out[~valid] = 0.0
    return out



def adjacent_cents(f0: np.ndarray) -> np.ndarray:
    x = np.asarray(f0, dtype=float)
    delta = np.full(len(x), np.nan, dtype=float)
    if len(x) < 2:
        return delta
    valid = (
        np.isfinite(x[1:])
        & np.isfinite(x[:-1])
        & (x[1:] > 0)
        & (x[:-1] > 0)
    )
    idx = np.nonzero(valid)[0] + 1
    delta[idx] = 1200.0 * np.log2(x[idx] / x[idx - 1])
    return delta


def _positive_context_median(values: np.ndarray, start: int, stop: int) -> float:
    """返回指定半开区间内正值中位数；无有效值时返回 NaN。"""
    x = np.asarray(values, dtype=float)
    segment = x[max(0, start): min(len(x), stop)]
    valid = segment[np.isfinite(segment) & (segment > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _absolute_cents(a_hz: float, b_hz: float) -> float:
    """两个正频率之间的绝对 cents 距离。"""
    if not (np.isfinite(a_hz) and a_hz > 0 and np.isfinite(b_hz) and b_hz > 0):
        return float("inf")
    return float(abs(1200.0 * np.log2(a_hz / b_hz)))



def repair_short_jump_bridges_simple(
    f0: np.ndarray,
    *,
    alpha: float = JUMP_BRIDGE_ALPHA,
    min_step_cents: float = JUMP_BRIDGE_MIN_STEP_CENTS,
    max_frames: int = JUMP_BRIDGE_MAX_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将短跳音中被模型连出的陡坡按帧数劈半，贴到即时左右边界。

    规则只有三步：
    1. 在连续非零轨迹中寻找至少两条连续陡边，单步绝对变化不低于
       ``min_step_cents``；
    2. 以紧邻陡边串的左右值为锚点，计算左右总跨度 D；陡边串中至少
       两步达到 ``max(min_step_cents, alpha * D)``；
    3. 陡边串中间最多 ``max_frames`` 帧，按帧数劈半：左半贴左锚点，
       右半贴右锚点。奇数中央帧贴到原值更接近的一侧。

    不跨零、不寻找远处平台、不看相对强度；这里只处理同方向的陡边串，
    来回振荡留给原轨迹，短八度小岛则交给后续八度规则。该步骤必须先于
    短八度小岛修正执行。
    """
    original = np.asarray(f0, dtype=float)
    repaired = original.copy()
    n = len(original)
    repair_mask = np.zeros(n, dtype=bool)
    repair_group = np.zeros(n, dtype=np.int32)
    repair_threshold = np.full(n, np.nan, dtype=float)
    if n < 4 or max_frames < 1:
        return repaired, repair_mask, repair_group, repair_threshold

    delta = adjacent_cents(original)
    steep = np.isfinite(delta) & (np.abs(delta) >= min_step_cents)
    group_id = 0

    i = 1
    while i < n:
        # 不能跨零；陡边 delta[i] 连接 i-1 与 i。
        if not steep[i]:
            i += 1
            continue

        edge_start = i
        edge_end = i
        while edge_end + 1 < n and steep[edge_end + 1]:
            edge_end += 1

        edge_count = edge_end - edge_start + 1
        left = edge_start - 1
        right = edge_end
        middle_start = edge_start
        middle_end = edge_end - 1
        middle_len = middle_end - middle_start + 1

        if (
            edge_count >= 2
            and 1 <= middle_len <= max_frames
            and left >= 0
            and right < n
            and np.isfinite(original[left]) and original[left] > 0
            and np.isfinite(original[right]) and original[right] > 0
            and np.all(np.isfinite(original[left:right + 1]))
            and np.all(original[left:right + 1] > 0)
        ):
            total_span = _absolute_cents(original[left], original[right])
            threshold = max(min_step_cents, alpha * total_span)
            local_steps = np.abs(delta[edge_start:edge_end + 1])

            local_signs = np.sign(delta[edge_start:edge_end + 1])
            same_direction = bool(np.all(local_signs > 0) or np.all(local_signs < 0))

            if int(np.sum(local_steps >= threshold)) >= 2 and same_direction:
                left_f0 = float(original[left])
                right_f0 = float(original[right])
                length = middle_len
                split = length // 2
                group_id += 1

                for offset, idx in enumerate(range(middle_start, middle_end + 1)):
                    if length % 2 == 1 and offset == split:
                        if _absolute_cents(original[idx], left_f0) <= _absolute_cents(original[idx], right_f0):
                            repaired[idx] = left_f0
                        else:
                            repaired[idx] = right_f0
                    elif offset < (length + 1) // 2:
                        repaired[idx] = left_f0
                    else:
                        repaired[idx] = right_f0

                    repair_mask[idx] = True
                    repair_group[idx] = group_id
                    repair_threshold[idx] = threshold

        i = edge_end + 1

    return repaired, repair_mask, repair_group, repair_threshold


def _is_octave_up_jump(cents: float) -> bool:
    return bool(
        np.isfinite(cents)
        and OCTAVE_REPAIR_MIN_CENTS <= cents <= OCTAVE_REPAIR_MAX_CENTS
    )


def _is_octave_down_jump(cents: float) -> bool:
    return bool(
        np.isfinite(cents)
        and -OCTAVE_REPAIR_MAX_CENTS <= cents <= -OCTAVE_REPAIR_MIN_CENTS
    )


def repair_short_octave_islands(
    f0: np.ndarray,
    max_frames: int = OCTAVE_REPAIR_MAX_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    只修正长度 1--15 帧的短高八度小岛，并保持小岛内部走势。

    检测范围固定为边界跳变 1100--1300 cents。支持三种形态：
    1. 封闭小岛：低轨迹 -> 高一个八度 -> 回到低轨迹；
    2. 音头小岛：非零段一开始就在高八度，随后向下跳回低轨迹；
    3. 音尾小岛：从低轨迹向上跳一个八度，并在15帧内结束。

    修正方法统一为 ``f / 2``，不会展平或插值，因此例如
    600->630->700 会变成 300->315->350，滑音形状仍然保留。
    超过15帧、边界不在指定范围、连接证据不足或与既有修正重叠时不修改。
    """
    original = np.asarray(f0, dtype=float)
    repaired = original.copy()
    n = len(original)
    repair_mask = np.zeros(n, dtype=bool)
    repair_group = np.zeros(n, dtype=np.int32)
    repair_type = np.full(n, "", dtype=object)
    if n == 0 or max_frames <= 0:
        return repaired, repair_mask, repair_group, repair_type

    delta = adjacent_cents(original)
    group_id = 0

    # 连续正值段之间的零区是天然边界；绝不跨零寻找或修正小岛。
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not (np.isfinite(original[i]) and original[i] > 0):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(original[j]) and original[j] > 0:
            j += 1
        runs.append((i, j - 1))
        i = j

    def apply_segment(start: int, end: int, kind: str) -> bool:
        nonlocal group_id
        if start < 0 or end >= n or end < start:
            return False
        length = end - start + 1
        if length < 1 or length > max_frames or np.any(repair_mask[start:end + 1]):
            return False
        segment = original[start:end + 1]
        if np.any(~np.isfinite(segment)) or np.any(segment <= 0):
            return False
        repaired[start:end + 1] = segment / 2.0
        repair_mask[start:end + 1] = True
        group_id += 1
        repair_group[start:end + 1] = group_id
        repair_type[start:end + 1] = kind
        return True

    # 优先处理证据最强的封闭小岛，避免被音头/音尾规则抢先占用。
    for run_start, run_end in runs:
        up_indices = [k for k in range(run_start + 1, run_end + 1) if _is_octave_up_jump(delta[k])]
        down_indices = [k for k in range(run_start + 1, run_end + 1) if _is_octave_down_jump(delta[k])]

        for start in up_indices:
            possible_exits = [k for k in down_indices if start < k <= start + max_frames]
            if not possible_exits:
                continue
            exit_index = possible_exits[0]
            end = exit_index - 1

            corrected_first = original[start] / 2.0
            corrected_last = original[end] / 2.0
            if (
                _absolute_cents(corrected_first, original[start - 1])
                <= OCTAVE_REPAIR_CONNECTION_CENTS
                and _absolute_cents(corrected_last, original[exit_index])
                <= OCTAVE_REPAIR_CONNECTION_CENTS
            ):
                apply_segment(start, end, "ENCLOSED")

    # 音头：连续正值段从高八度开始，15帧内向下跳回低轨迹。
    for run_start, run_end in runs:
        if repair_mask[run_start]:
            continue
        latest_exit = min(run_end, run_start + max_frames)
        exits = [
            k for k in range(run_start + 1, latest_exit + 1)
            if _is_octave_down_jump(delta[k])
        ]
        if not exits:
            continue
        exit_index = exits[0]
        end = exit_index - 1
        corrected_last = original[end] / 2.0
        if (
            _absolute_cents(corrected_last, original[exit_index])
            <= OCTAVE_REPAIR_CONNECTION_CENTS
        ):
            apply_segment(run_start, end, "PREFIX")

    # 音尾：从低轨迹向上跳一个八度，并在15帧内结束。
    for run_start, run_end in runs:
        earliest_entry = max(run_start + 1, run_end - max_frames + 1)
        entries = [
            k for k in range(earliest_entry, run_end + 1)
            if _is_octave_up_jump(delta[k])
        ]
        if not entries:
            continue
        start = entries[-1]
        if np.any(repair_mask[start:run_end + 1]):
            continue
        corrected_first = original[start] / 2.0
        if (
            _absolute_cents(corrected_first, original[start - 1])
            <= OCTAVE_REPAIR_CONNECTION_CENTS
        ):
            apply_segment(start, run_end, "SUFFIX")

    return repaired, repair_mask, repair_group, repair_type


def fuse_pitch_tracks(
    crepe_hz: np.ndarray,
    periodicity: np.ndarray,
    dswipe_hz: np.ndarray,
    db: np.ndarray,
) -> pd.DataFrame:
    """按已确认参数融合两条 F0 轨迹，并返回最终轨迹与诊断信息。"""
    crepe_raw = np.asarray(crepe_hz, dtype=float)
    peri = np.asarray(periodicity, dtype=float)
    dswipe_raw = np.asarray(dswipe_hz, dtype=float)
    db_arr = np.asarray(db, dtype=float)

    n = min(len(crepe_raw), len(peri), len(dswipe_raw), len(db_arr))
    crepe_raw = crepe_raw[:n]
    peri = peri[:n]
    dswipe_raw = dswipe_raw[:n]
    db_arr = db_arr[:n]

    crepe = centered_median5(crepe_raw)
    dswipe = segmented_dswipe_median(dswipe_raw, db_arr)

    db_invalid = ~np.isfinite(db_arr) | (db_arr < DB_HARD_ZERO)

    dswipe_valid = (
        np.isfinite(dswipe)
        & (dswipe > 0)
        & (dswipe <= HARD_MAX_F0_HZ)
    )

    # periodicity < 0.01 直接清零；不做低置信 dSWIPE 回填。
    hard_invalid = (
        db_invalid
        | ~np.isfinite(peri)
        | (peri < PERIODICITY_HARD_ZERO)
    )

    crepe_valid = (
        np.isfinite(crepe)
        & (crepe > 0)
        & (crepe <= CREPE_FMAX)
        & (peri >= PERIODICITY_CREPE_MIN)
    )
    # dSWIPE 已先做 7 帧保零分段中值；3 帧中位数只用于让频段权重更稳定。
    control = positive_median3(dswipe)
    control = np.where(control > 0, control, np.where(crepe_valid, crepe, 0.0))

    final = np.zeros(n, dtype=float)
    crepe_weight = np.zeros(n, dtype=float)
    source = np.full(n, "INVALID", dtype=object)

    # 0.01 <= periodicity < 0.03：CREPE 禁止参与，使用 dSWIPE 中值轨迹。
    d_only_low_peri = (
        ~hard_invalid
        & (peri < PERIODICITY_CREPE_MIN)
        & dswipe_valid
    )
    final[d_only_low_peri] = dswipe[d_only_low_peri]
    source[d_only_low_peri] = "DSWIPE_LOW_PERIODICITY"

    both = ~hard_invalid & crepe_valid & dswipe_valid

    low_main = both & (control <= LOW_DSWIPE_END_HZ)
    final[low_main] = dswipe[low_main]
    source[low_main] = "DSWIPE_LOW"

    low_transition = both & (control > LOW_DSWIPE_END_HZ) & (control < LOW_CREPE_START_HZ)
    if np.any(low_transition):
        u = (control[low_transition] - LOW_DSWIPE_END_HZ) / (
            LOW_CREPE_START_HZ - LOW_DSWIPE_END_HZ
        )
        w_crepe = smoothstep01(u)
        final[low_transition] = log_blend(
            dswipe[low_transition], crepe[low_transition], w_crepe
        )
        crepe_weight[low_transition] = w_crepe
        source[low_transition] = "BLEND_LOW"

    mid_main = both & (control >= LOW_CREPE_START_HZ) & (control <= HIGH_CREPE_END_HZ)
    final[mid_main] = crepe[mid_main]
    crepe_weight[mid_main] = 1.0
    source[mid_main] = "CREPE_MID"

    high_transition = both & (control > HIGH_CREPE_END_HZ) & (control < HIGH_DSWIPE_START_HZ)
    if np.any(high_transition):
        u = (control[high_transition] - HIGH_CREPE_END_HZ) / (
            HIGH_DSWIPE_START_HZ - HIGH_CREPE_END_HZ
        )
        w_dswipe = smoothstep01(u)
        final[high_transition] = log_blend(
            crepe[high_transition], dswipe[high_transition], w_dswipe
        )
        crepe_weight[high_transition] = 1.0 - w_dswipe
        source[high_transition] = "BLEND_HIGH"

    high_main = both & (control >= HIGH_DSWIPE_START_HZ)
    final[high_main] = dswipe[high_main]
    source[high_main] = "DSWIPE_HIGH"

    # 只有某一个来源有效时才 fallback；固定权重不因算法差异而临时改变。
    crepe_only = ~hard_invalid & crepe_valid & ~dswipe_valid
    final[crepe_only] = crepe[crepe_only]
    crepe_weight[crepe_only] = 1.0
    source[crepe_only] = "CREPE_FALLBACK"

    dswipe_only = (
        ~hard_invalid
        & dswipe_valid
        & ~crepe_valid
        & (peri >= PERIODICITY_CREPE_MIN)
    )
    final[dswipe_only] = dswipe[dswipe_only]
    source[dswipe_only] = "DSWIPE_FALLBACK"

    # 最终硬上限。
    final[(~np.isfinite(final)) | (final < 0) | (final > HARD_MAX_F0_HZ)] = 0.0

    # 永久无效掩码最后再执行一次，保证任何 fallback 都不能把它恢复。
    final[hard_invalid] = 0.0
    source[hard_invalid] = "HARD_INVALID"
    crepe_weight[hard_invalid] = 0.0

    # 先将短跳音中被模型连出的陡坡劈半并贴到即时左右边界。
    (
        final,
        jump_bridge_repair,
        jump_bridge_group,
        jump_bridge_threshold,
    ) = repair_short_jump_bridges_simple(final)

    # 再处理残留的 1--15 帧短高八度小岛；整段除以2，保留内部走势。
    (
        final,
        octave_short_repair,
        octave_repair_group,
        octave_repair_type,
    ) = repair_short_octave_islands(final)

    delta = adjacent_cents(final)
    jump_flag = np.isfinite(delta) & (np.abs(delta) >= JUMP_FLAG_CENTS)

    return pd.DataFrame(
        {
            FINAL_COL: final,
            "CREPE中值5(Hz)": crepe,
            "dSWIPE中值7(Hz)": dswipe,
            "频段控制频率(Hz)": control,
            "CREPE权重": crepe_weight,
            "选用来源": source,
            "相邻变化(cents)": delta,
            "jump_flag": jump_flag,
            "jump_bridge_repair": jump_bridge_repair,
            "jump_bridge_group": jump_bridge_group,
            "jump_bridge_threshold(cents)": jump_bridge_threshold,
            "octave_short_repair": octave_short_repair,
            "octave_repair_group": octave_repair_group,
            "octave_repair_type": octave_repair_type,
            "hard_invalid": hard_invalid,
        }
    )


# =========================
# 主流程
# =========================
def main() -> None:
    wav_path, stem = pick_audio()
    print(f"🎵 使用音频：{wav_path.name}")

    db = load_reference_db(stem)
    crepe, peri = run_crepe(wav_path)
    dswipe_df = run_dswipe(wav_path)

    n = min(len(crepe), len(peri), len(db))
    if n == 0:
        raise SystemExit("❌ CREPE 或音量数据为空")

    crepe = crepe[:n]
    peri = peri[:n]
    db = db[:n]
    times = np.arange(n, dtype=float) * HOP_SECONDS
    dswipe = align_dswipe(times, dswipe_df)

    fused = fuse_pitch_tracks(crepe, peri, dswipe, db)
    n = min(n, len(fused))

    # 前六列为稳定接口；诊断列统一放到其后。
    output = pd.DataFrame(
        {
            TIME_COL: np.round(times[:n], 3),
            CREPE_COL: np.round(crepe[:n], 2),
            PERIODICITY_COL: np.round(peri[:n], 4),
            DB_COL: np.round(db[:n], 2),
            DSWIPE_COL: np.round(dswipe[:n], 2),
            FINAL_COL: np.round(fused[FINAL_COL].to_numpy()[:n], 2),
            "CREPE中值5(Hz)": np.round(fused["CREPE中值5(Hz)"].to_numpy()[:n], 2),
            "dSWIPE中值7(Hz)": np.round(fused["dSWIPE中值7(Hz)"].to_numpy()[:n], 2),
            "频段控制频率(Hz)": np.round(fused["频段控制频率(Hz)"].to_numpy()[:n], 2),
            "CREPE权重": np.round(fused["CREPE权重"].to_numpy()[:n], 4),
            "选用来源": fused["选用来源"].to_numpy()[:n],
            "相邻变化(cents)": np.round(fused["相邻变化(cents)"].to_numpy()[:n], 2),
            "jump_flag": fused["jump_flag"].to_numpy()[:n],
            "jump_bridge_repair": fused["jump_bridge_repair"].to_numpy()[:n],
            "jump_bridge_group": fused["jump_bridge_group"].to_numpy()[:n],
            "jump_bridge_threshold(cents)": np.round(
                fused["jump_bridge_threshold(cents)"].to_numpy()[:n], 2
            ),
            "octave_short_repair": fused["octave_short_repair"].to_numpy()[:n],
            "octave_repair_group": fused["octave_repair_group"].to_numpy()[:n],
            "octave_repair_type": fused["octave_repair_type"].to_numpy()[:n],
            "hard_invalid": fused["hard_invalid"].to_numpy()[:n],
        }
    )

    out_csv = BASE_DIR / f"{stem}_CREPE_plain_raw.csv"
    output.to_csv(out_csv, index=False, encoding="utf-8-sig")

    final_f0 = output[FINAL_COL].to_numpy(dtype=float)
    hard_count = int(output["hard_invalid"].sum())
    jump_bridge_count = int(output["jump_bridge_repair"].sum())
    octave_repair_count = int(output["octave_short_repair"].sum())
    jump_count = int(output["jump_flag"].sum())
    print(f"✅ 已保存：{out_csv.name}")
    print(f"✅ 帧数：{len(output)} | 最终非零：{int(np.sum(final_f0 > 0))}")
    print(f"✅ CREPE 已执行 5 帧居中中值；原始 CREPE 列保持不变")
    print(f"✅ dSWIPE 已执行 {DSWIPE_MEDIAN_WINDOW} 帧保零分段中值；原始 dSWIPE 列保持不变")
    print(f"✅ periodicity<0.01 或 dB<5 的硬清零：{hard_count} 帧")
    print(
        f"✅ 短跳音伪连接修正：{jump_bridge_count} 帧 "
        f"（alpha={JUMP_BRIDGE_ALPHA:.2f}，单步下限{JUMP_BRIDGE_MIN_STEP_CENTS:.0f}c，"
        f"≤{JUMP_BRIDGE_MAX_FRAMES}帧；中间劈半贴即时左右值）"
    )
    print(
        f"✅ 随后短高八度小岛修正：{octave_repair_count} 帧 "
        f"（{OCTAVE_REPAIR_MIN_CENTS:.0f}--{OCTAVE_REPAIR_MAX_CENTS:.0f} cents，"
        f"≤{OCTAVE_REPAIR_MAX_FRAMES}帧，整段/2）"
    )
    print(f"✅ 相邻变化 >= {JUMP_FLAG_CENTS:.0f} cents：{jump_count} 帧（仅标记，不平滑）")
    print("✅ 未设置 250/280Hz 低频硬清零；未执行长零回填；超过15帧的八度片段不修正")


if __name__ == "__main__":
    main()
