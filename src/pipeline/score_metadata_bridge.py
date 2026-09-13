from __future__ import annotations

import contextlib
import os
import runpy
from pathlib import Path
from typing import Iterable

import pandas as pd

from alignment_contract import choose_onset_file

SCORE_ID_COL = "score_note_id"
ALIGN_FLAG_COL = "参与起音对齐"
LOGICAL_DUR_COL = "逻辑持续时间(s)"
DURATION_COL = "持续时间(s)"

# These columns are score semantics and should survive every downstream stage.
SCORE_METADATA_COLUMNS = [
    "score_note_id",
    "参与起音对齐",
    "对齐组ID",
    "对齐锚点ID",
    "逻辑持续时间(s)",
    "逻辑持续时间(拍)",
    "逻辑结束时间(s)",
    "小节号",
    "声部",
    "谱面起点(拍)",
    "谱面时值(拍)",
    "强弱记号",
    "当前强弱",
    "强弱变化",
    "延音线状态",
    "连音线状态",
    "连音线编号",
    "滑音线状态",
    "滑音线编号",
    "谱面文字",
    "其他记号",
    "拍号",
    "调号升降数",
]


def _single_file(base_dir: Path, pattern: str, description: str) -> Path:
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到{description}: {pattern}")
    if len(files) > 1:
        names = ", ".join(p.name for p in files)
        raise RuntimeError(f"检测到多个{description}，请清理目录后重试: {names}")
    return files[0]


def find_score_file(base_dir: Path) -> Path:
    return _single_file(base_dir, "*_score_final.csv", "score_final")


def find_onset_file(base_dir: Path) -> Path:
    return choose_onset_file(base_dir)


def load_full_and_alignment_score(base_dir: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    score_path = find_score_file(base_dir)
    full = pd.read_csv(score_path)

    if SCORE_ID_COL not in full.columns:
        # Old score table: preserve old behavior rather than breaking the pipeline.
        full[SCORE_ID_COL] = [f"legacy_score_{i:06d}" for i in range(len(full))]

    if full[SCORE_ID_COL].astype(str).duplicated().any():
        dup = full.loc[full[SCORE_ID_COL].astype(str).duplicated(), SCORE_ID_COL].tolist()
        raise ValueError(f"score_note_id 重复: {dup[:10]}")

    if ALIGN_FLAG_COL in full.columns:
        flag = full[ALIGN_FLAG_COL].fillna("是").astype(str).str.strip()
        align = full[flag != "否"].copy()
    else:
        align = full.copy()

    align = align.reset_index(drop=True)

    # Tie chain anchors use the summed logical duration during alignment.
    if LOGICAL_DUR_COL in align.columns and DURATION_COL in align.columns:
        logical = pd.to_numeric(align[LOGICAL_DUR_COL], errors="coerce")
        original = pd.to_numeric(align[DURATION_COL], errors="coerce")
        use = logical.notna() & (logical > 0)
        align.loc[use, DURATION_COL] = logical[use]
        align.loc[~use, DURATION_COL] = original[~use]

    return score_path, full, align


@contextlib.contextmanager
def temporary_score_file(score_path: Path, alignment_score: pd.DataFrame):
    original_bytes = score_path.read_bytes()
    try:
        alignment_score.to_csv(score_path, index=False, encoding="utf-8-sig")
        yield
    finally:
        score_path.write_bytes(original_bytes)


def run_legacy_script(base_dir: Path, legacy_filename: str) -> None:
    legacy_path = base_dir / legacy_filename
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"未找到正式算法实现 {legacy_filename}。"
        )
    runpy.run_path(str(legacy_path), run_name="__main__")


def validate_id_sequence(path: Path, alignment_score: pd.DataFrame, description: str) -> None:
    df = pd.read_csv(path)
    expected = alignment_score[SCORE_ID_COL].astype(str).tolist()

    if SCORE_ID_COL not in df.columns:
        if len(df) != len(expected):
            raise ValueError(
                f"{description} 没有 score_note_id，且行数与对齐 score 不同: "
                f"{len(df)} != {len(expected)}"
            )
        return

    actual = df[SCORE_ID_COL].astype(str).tolist()
    if actual != expected:
        raise ValueError(
            f"{description} 的 score_note_id 顺序与 score_final 不一致，停止继续运行。"
        )


def _metadata_frame(alignment_score: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SCORE_METADATA_COLUMNS if c in alignment_score.columns]
    return alignment_score[cols].reset_index(drop=True).copy()


def merge_score_metadata(
    output_path: Path,
    alignment_score: pd.DataFrame,
    *,
    allow_black_tail: bool = False,
) -> pd.DataFrame:
    """Append score metadata to a downstream CSV without overwriting algorithm columns."""
    out = pd.read_csv(output_path)
    meta = _metadata_frame(alignment_score)

    if allow_black_tail:
        black_mask = pd.Series(False, index=out.index)
        for col in ["真正起音来源", "真正起音颜色"]:
            if col in out.columns:
                black_mask |= out[col].fillna("").astype(str).str.strip().str.lower().eq("black")
        if "score_idx" in out.columns:
            black_mask |= out["score_idx"].fillna("").astype(str).str.strip().str.lower().eq("black")
        main_indices = out.index[~black_mask].tolist()
    else:
        main_indices = out.index.tolist()

    if len(main_indices) != len(meta):
        raise ValueError(
            f"无法透传 score 元数据：{output_path.name} 主体行数={len(main_indices)}，"
            f"对齐 score 行数={len(meta)}。"
        )

    # score_note_id is the identity contract. Always set it from score metadata.
    for col in meta.columns:
        values = meta[col].tolist()
        if col not in out.columns:
            out[col] = pd.NA
        for pos, row_index in enumerate(main_indices):
            if col == SCORE_ID_COL or pd.isna(out.at[row_index, col]) or str(out.at[row_index, col]).strip() == "":
                out.at[row_index, col] = values[pos]

    output_path.write_text(out.to_csv(index=False), encoding="utf-8-sig")
    return out


def find_stage_output(base_dir: Path, tag: str) -> Path:
    return _single_file(
        base_dir,
        f"*_{tag}_对齐结果_score主导.csv",
        f"{tag} 对齐结果",
    )




def run_stage_wrapper(stage: str, legacy_filename: str) -> Path:
    code_dir = Path(__file__).resolve().parent
    base_dir = Path(os.environ.get("ERHU_WORK_DIR", str(code_dir))).resolve()
    score_path, _full_score, alignment_score = load_full_and_alignment_score(base_dir)

    previous = None
    if stage == "DP2":
        previous = find_stage_output(base_dir, "DP1")
    elif stage == "DP3":
        previous = find_stage_output(base_dir, "DP2")

    if previous is not None:
        validate_id_sequence(previous, alignment_score, previous.name)

    with temporary_score_file(score_path, alignment_score):
        run_legacy_script(code_dir, legacy_filename)

    output = find_stage_output(base_dir, stage)
    merge_score_metadata(output, alignment_score)
    validate_id_sequence(output, alignment_score, output.name)
    print(f"✅ {stage} score_note_id / score metadata字段已透传: {output.name}")
    return output


