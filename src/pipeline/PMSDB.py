import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 自动获取音频文件
# ==========================================
BASE_DIR = os.path.abspath(
    os.environ.get("ERHU_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
)
wav_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith('.wav')]

if not wav_files:
    print("❌ 错误: 未找到 .wav 文件")
    exit()

AUDIO_NAME = wav_files[0]
AUDIO_PATH = os.path.join(BASE_DIR, AUDIO_NAME)
FILE_STEM = os.path.splitext(AUDIO_NAME)[0]

def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    mx = float(np.max(x))
    if mx <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x / mx).astype(np.float32)

def run_5ms_energy_analysis():
    print(f"🚀 开启 5ms 高精度分析: {AUDIO_NAME}")
    
    # 保持 16kHz 采样率对齐 CREPE / onset
    y, sr = librosa.load(AUDIO_PATH, sr=16000)

    # ==========================================
    # 2. 参数设置
    # ==========================================
    HOP_LENGTH = 80     # 5ms 步长 (16000 * 0.005)
    FRAME_LENGTH = 256  # 16ms 窗口

    # ==========================================
    # 3. RMS / dB
    # ==========================================
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    db_raw = librosa.amplitude_to_db(rms, ref=np.max)

    # 以最小声音为 0 基准
    noise_floor = np.min(db_raw)
    db_relative = db_raw - noise_floor

    # ==========================================
    # 4. onset strength
    # ==========================================
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    onset_env_norm = normalize01(onset_env)

    # ==========================================
    # 5. 时间轴
    # ==========================================
    n = min(len(rms), len(onset_env), int(np.ceil(len(y) / HOP_LENGTH)))
    rms = rms[:n]
    db_relative = db_relative[:n]
    onset_env = onset_env[:n]
    onset_env_norm = onset_env_norm[:n]

    times = [i * (HOP_LENGTH / sr) for i in range(n)]

    # ==========================================
    # 6. 输出三个表格
    # ==========================================
    # 表 1: RMS 能量表
    df_rms = pd.DataFrame({
        '时间(s)': times,
        'RMS能量': np.round(rms, 6)
    })
    rms_csv = os.path.join(BASE_DIR, f"{FILE_STEM}_RMS能量_5ms.csv")
    df_rms.to_csv(rms_csv, index=False, encoding="utf-8-sig")

    # 表 2: 相对动态表
    df_db = pd.DataFrame({
        '时间(s)': times,
        '相对强度(dB)': np.round(db_relative, 2)
    })
    db_csv = os.path.join(BASE_DIR, f"{FILE_STEM}_相对动态_5ms.csv")
    df_db.to_csv(db_csv, index=False, encoding="utf-8-sig")

    # 表 3: onset 强度表
    df_onset = pd.DataFrame({
        '时间(s)': times,
        'onset_strength': np.round(onset_env, 6),
        'onset_strength_norm': np.round(onset_env_norm, 6)
    })
    onset_csv = os.path.join(BASE_DIR, f"{FILE_STEM}_onset强度_5ms.csv")
    df_onset.to_csv(onset_csv, index=False, encoding="utf-8-sig")

    print("✅ 数据表保存完毕 (每秒 200 行)")
    print(f"   - {os.path.basename(rms_csv)}")
    print(f"   - {os.path.basename(db_csv)}")
    print(f"   - {os.path.basename(onset_csv)}")

    # ==========================================
    # 7. 生成图表
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 图 1: RMS 能量
    plt.figure(figsize=(15, 5))
    plt.plot(times, rms, color='teal', linewidth=0.8)
    plt.fill_between(times, rms, color='teal', alpha=0.1)
    plt.title(f'RMS 能量包络 (5ms 高精度) - {AUDIO_NAME}')
    plt.xlabel('时间 (s)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{FILE_STEM}_RMS_5ms.png"), dpi=150)

    # 图 2: 相对动态 dB
    plt.figure(figsize=(15, 5))
    plt.plot(times, db_relative, color='crimson', linewidth=0.8)
    plt.fill_between(times, db_relative, color='crimson', alpha=0.1)
    plt.title(f'相对响度曲线 (以底噪为0) - {AUDIO_NAME}')
    plt.xlabel('时间 (s)')
    plt.ylabel('增量 (dB)')
    plt.ylim(0, np.max(db_relative) + 10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{FILE_STEM}_dB_5ms.png"), dpi=150)

    # 图 3: onset strength
    plt.figure(figsize=(15, 5))
    plt.plot(times, onset_env, color='royalblue', linewidth=0.8, label='onset_strength')
    plt.plot(times, onset_env_norm, color='orange', linewidth=0.8, alpha=0.8, label='onset_strength_norm')
    plt.title(f'Onset 强度曲线 (5ms) - {AUDIO_NAME}')
    plt.xlabel('时间 (s)')
    plt.ylabel('强度')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{FILE_STEM}_onset_strength_5ms.png"), dpi=150)

    print("🖼️ 图表生成完毕。")
    print(f"📊 动态范围: {np.max(db_relative):.2f} dB")
    print(f"📊 onset_strength max: {np.max(onset_env):.6f}")
    # plt.show()

if __name__ == "__main__":
    run_5ms_energy_analysis()
