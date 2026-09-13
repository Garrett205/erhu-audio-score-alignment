#!/usr/bin/env python3
"""Evaluate the frozen Erhu pitch front end on all CCOM-HuQin references.

This utility deliberately sits outside the formal six-piece workflow.  It can:

1. evaluate existing per-piece F0 CSV files; or
2. run only the frozen ``PMSDB.py -> pitch.py`` front end on 21 WAV files and
   then evaluate the generated F0 tracks.

It never edits the pitch front end, alignment code, frozen artifacts, or the
formal three-piece RPA50 files.  Write results to a new output directory.

An optional QC-exclusion manifest can restrict evaluation to a documented
subset without deleting or altering any source result.  The manifest must have
a ``piece`` column and may record a ``reason`` column; both the manifest and
the resolved exclusions are copied to the new output directory.

Metrics
-------
Every reference frame is matched to the nearest predicted frame within
``tolerance_ms``. The script reports two separate views: strict RPA50 on
reference-voiced frames (missing/zero predictions fail), and all-frame joint
F0+voicing accuracy. In the latter, a reference-silent frame is correct only
when the prediction is also unvoiced; a reference-voiced frame must satisfy
the same 50-cent criterion. Voicing false alarms, recall, and precision are
reported separately. Per-piece macro and pooled micro values remain separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REFERENCE_SUFFIX = "-pitch.csv"
WAV_SUFFIXES = {".wav"}
TIME_ALIASES = (
    "time_s",
    "time",
    "时间(s)",
    "time_raw_in_pitch1(s)",
    "time_raw",
    "time_stretched(s)",
)
F0_ALIASES = (
    "onset频率(Hz)",  # frozen pitch.py final output
    "final_f0_hz",
    "pitch(Hz)",
    "f0_hz",
    "f0",
    "pitch",
    "frequency",
)


@dataclass(frozen=True)
class Track:
    time_s: np.ndarray
    f0_hz: np.ndarray
    time_column: str
    f0_column: str


def canonical(text: str) -> str:
    """Stable ASCII identifier used only for explicit filename matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Cannot decode {path}: {'; '.join(errors)}")


def select_column(columns: Iterable[object], aliases: Iterable[str], label: str, path: Path) -> str:
    labels = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        found = labels.get(alias.strip().lower())
        if found is not None:
            return found
    raise ValueError(
        f"{path} has no supported {label} column. Columns: "
        f"{[str(column) for column in columns]}"
    )


def load_track(path: Path, *, time_column: str | None = None, f0_column: str | None = None) -> Track:
    frame = read_csv_robust(path)
    if frame.empty:
        raise ValueError(f"Empty CSV: {path}")
    time_column = time_column or select_column(frame.columns, TIME_ALIASES, "time", path)
    f0_column = f0_column or select_column(frame.columns, F0_ALIASES, "F0", path)
    if time_column not in frame.columns or f0_column not in frame.columns:
        raise ValueError(f"Requested columns are absent in {path}")
    time_s = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    f0_hz = pd.to_numeric(frame[f0_column], errors="coerce").to_numpy(float)
    keep = np.isfinite(time_s)
    if not np.any(keep):
        raise ValueError(f"No finite timestamps in {path}")
    order = np.argsort(time_s[keep], kind="stable")
    return Track(
        time_s=time_s[keep][order],
        f0_hz=f0_hz[keep][order],
        time_column=time_column,
        f0_column=f0_column,
    )


def reference_pieces(reference_root: Path) -> dict[str, Path]:
    pieces: dict[str, Path] = {}
    for path in sorted(reference_root.rglob(f"*{REFERENCE_SUFFIX}")):
        name = path.name[: -len(REFERENCE_SUFFIX)]
        if path.parent.name != name:
            continue
        if name in pieces:
            raise ValueError(f"Duplicate reference for {name}: {pieces[name]} and {path}")
        pieces[name] = path
    if not pieces:
        raise ValueError(f"No CCOM reference files named *{REFERENCE_SUFFIX} under {reference_root}")
    return pieces


def read_manifest(path: Path, required_column: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError(f"Empty manifest: {path}")
    if "piece" not in rows[0] or required_column not in rows[0]:
        raise ValueError(f"{path} must contain columns: piece,{required_column}")
    values: dict[str, dict[str, str]] = {}
    for row in rows:
        piece = (row.get("piece") or "").strip()
        value = (row.get(required_column) or "").strip()
        if not piece or not value:
            raise ValueError(f"Invalid manifest row in {path}: {row}")
        if piece in values:
            raise ValueError(f"Duplicate piece in manifest {path}: {piece}")
        values[piece] = {key: (value or "").strip() for key, value in row.items() if key}
    return values


def resolve_manifest_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def discover_exact_files(root: Path, pieces: Iterable[str], suffixes: set[str]) -> dict[str, Path]:
    by_key: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            by_key.setdefault(canonical(path.stem), []).append(path.resolve())
    result: dict[str, Path] = {}
    for piece in pieces:
        matches = by_key.get(canonical(piece), [])
        if len(matches) == 1:
            result[piece] = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous input for {piece}: {[str(match) for match in matches]}. "
                "Use a manifest to select one file explicitly."
            )
    return result


def prediction_candidates(root: Path, reference_root: Path, pieces: Iterable[str]) -> dict[str, Path]:
    """Find exact-name CSV candidates without accidentally reusing references."""
    ref_root = reference_root.resolve()
    by_key: dict[str, list[Path]] = {}
    for path in root.rglob("*.csv"):
        resolved = path.resolve()
        if ref_root == resolved.parent or ref_root in resolved.parents:
            continue
        stem = path.stem
        for ending in ("_CREPE_plain_raw", "_pitch2", "-pitch2", "_pitch", "-pitch"):
            if stem.lower().endswith(ending.lower()):
                stem = stem[: -len(ending)]
                break
        by_key.setdefault(canonical(stem), []).append(resolved)
    result: dict[str, Path] = {}
    for piece in pieces:
        matches = by_key.get(canonical(piece), [])
        if len(matches) == 1:
            result[piece] = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous prediction for {piece}: {[str(match) for match in matches]}. "
                "Use --prediction-manifest."
            )
    return result


def nearest_prediction(ref_time: np.ndarray, pred_time: np.ndarray, pred_f0: np.ndarray, tolerance_s: float) -> np.ndarray:
    """Nearest-frame association, with zero denoting no prediction in tolerance."""
    right = np.searchsorted(pred_time, ref_time, side="left")
    left = np.clip(right - 1, 0, len(pred_time) - 1)
    right = np.clip(right, 0, len(pred_time) - 1)
    left_distance = np.abs(ref_time - pred_time[left])
    right_distance = np.abs(pred_time[right] - ref_time)
    indices = np.where(left_distance <= right_distance, left, right)
    distance = np.minimum(left_distance, right_distance)
    estimated = pred_f0[indices].astype(float, copy=True)
    estimated[distance > tolerance_s] = 0.0
    estimated[~np.isfinite(estimated)] = 0.0
    return estimated


def cents_error(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    result = np.full(len(ref), np.nan, dtype=float)
    valid = np.isfinite(est) & np.isfinite(ref) & (est > 0) & (ref > 0)
    result[valid] = 1200.0 * np.log2(est[valid] / ref[valid])
    return result


def score_piece(piece: str, reference_path: Path, prediction_path: Path, args: argparse.Namespace, overrides: dict[str, str]) -> dict[str, object]:
    reference = load_track(reference_path)
    prediction = load_track(
        prediction_path,
        time_column=overrides.get("time_column") or None,
        f0_column=overrides.get("f0_column") or None,
    )
    voiced = np.isfinite(reference.f0_hz) & (reference.f0_hz > 0) & (reference.f0_hz <= args.max_ref_hz)
    unvoiced = np.isfinite(reference.f0_hz) & (reference.f0_hz <= 0)
    valid_reference = voiced | unvoiced
    estimate = nearest_prediction(reference.time_s, prediction.time_s, prediction.f0_hz, args.tolerance_ms / 1000.0)
    covered = voiced & (estimate > 0) & np.isfinite(estimate)
    error = np.abs(cents_error(estimate, reference.f0_hz))
    correct = covered & (error <= 50.0)
    false_alarm = unvoiced & (estimate > 0) & np.isfinite(estimate)
    true_unvoiced = unvoiced & ~false_alarm
    predicted_voiced = covered | false_alarm
    joint_correct = correct | true_unvoiced
    count = int(np.sum(voiced))
    covered_count = int(np.sum(covered))
    correct_count = int(np.sum(correct))
    unvoiced_count = int(np.sum(unvoiced))
    false_alarm_count = int(np.sum(false_alarm))
    total_count = int(np.sum(valid_reference))
    true_unvoiced_count = int(np.sum(true_unvoiced))
    predicted_voiced_count = int(np.sum(predicted_voiced))
    joint_correct_count = int(np.sum(joint_correct))
    covered_error = error[covered]
    return {
        "piece": piece,
        "reference_csv": str(reference_path),
        "prediction_csv": str(prediction_path),
        "reference_time_column": reference.time_column,
        "reference_f0_column": reference.f0_column,
        "prediction_time_column": prediction.time_column,
        "prediction_f0_column": prediction.f0_column,
        "reference_voiced_frames": count,
        "reference_total_frames": total_count,
        "covered_frames": covered_count,
        "correct_rpa50_frames": correct_count,
        "coverage_pct": 100.0 * covered_count / count if count else np.nan,
        "mutually_voiced_rpa50_pct": 100.0 * correct_count / covered_count if covered_count else np.nan,
        "rpa50_pct": 100.0 * correct_count / count if count else np.nan,
        "reference_unvoiced_frames": unvoiced_count,
        "true_unvoiced_frames": true_unvoiced_count,
        "voicing_false_alarm_frames": false_alarm_count,
        "voicing_false_alarm_pct": 100.0 * false_alarm_count / unvoiced_count if unvoiced_count else np.nan,
        "voicing_precision_pct": 100.0 * covered_count / predicted_voiced_count if predicted_voiced_count else np.nan,
        "joint_f0_voicing_correct_frames": joint_correct_count,
        "joint_f0_voicing_accuracy_pct": 100.0 * joint_correct_count / total_count if total_count else np.nan,
        "mae_cents_covered": float(np.mean(covered_error)) if covered_error.size else np.nan,
        "median_cents_covered": float(np.median(covered_error)) if covered_error.size else np.nan,
        "prediction_frames": len(prediction.time_s),
    }


def run_frozen_frontend(audio_paths: dict[str, Path], output_dir: Path, pipeline_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    pmsdb = pipeline_dir / "PMSDB.py"
    pitch = pipeline_dir / "pitch.py"
    for required in (pmsdb, pitch):
        if not required.is_file():
            raise FileNotFoundError(f"Frozen front-end file not found: {required}")

    prediction_root = output_dir / "frontend_outputs"
    prediction_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for index, (piece, audio_path) in enumerate(sorted(audio_paths.items()), start=1):
        if audio_path.suffix.lower() not in WAV_SUFFIXES:
            raise ValueError(f"{piece}: frozen pipeline accepts WAV only, got {audio_path}")
        work_dir = prediction_root / piece
        output_csv = work_dir / f"{piece}_CREPE_plain_raw.csv"
        if output_csv.is_file() and args.reuse_frontend_outputs:
            result[piece] = output_csv.resolve()
            print(f"[{index:02d}] reuse {piece}: {output_csv}", flush=True)
            continue
        if work_dir.exists() and any(work_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to reuse non-empty work directory {work_dir}. "
                "Use --reuse-frontend-outputs only when its final CSV already exists, "
                "or choose a new --output-dir."
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        staged_wav = work_dir / f"{piece}.wav"
        shutil.copy2(audio_path, staged_wav)
        env = os.environ.copy()
        env["ERHU_WORK_DIR"] = str(work_dir.resolve())
        env.setdefault("PYTHONUTF8", "1")
        print(f"[{index:02d}/{len(audio_paths):02d}] frozen pitch front end: {piece}", flush=True)
        subprocess.run([args.python, str(pmsdb)], cwd=pipeline_dir, env=env, check=True)
        subprocess.run([args.python, str(pitch)], cwd=pipeline_dir, env=env, check=True)
        generated = sorted(work_dir.glob("*_CREPE_plain_raw.csv"))
        if generated != [output_csv]:
            raise RuntimeError(f"{piece}: expected {output_csv.name}, found {[path.name for path in generated]}")
        result[piece] = output_csv.resolve()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, required=True, help="Root containing the 21 <piece>/<piece>-pitch.csv folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this evaluation only.")
    parser.add_argument("--prediction-root", type=Path, help="Root containing existing predicted F0 CSV files.")
    parser.add_argument("--prediction-manifest", type=Path, help="CSV: piece,prediction_csv[,time_column,f0_column].")
    parser.add_argument("--exclude-manifest", type=Path, help="Optional CSV: piece[,reason]. Excludes documented QC-incompatible pieces from this evaluation run.")
    parser.add_argument("--run-frozen-frontend", action="store_true", help="Run only frozen PMSDB.py -> pitch.py on the supplied WAVs before evaluation.")
    parser.add_argument("--audio-root", type=Path, help="Root containing exactly named 21 WAV files; needed with --run-frozen-frontend unless --audio-manifest is used.")
    parser.add_argument("--audio-manifest", type=Path, help="CSV: piece,audio_path; resolves relative paths from the manifest directory.")
    parser.add_argument("--pipeline-dir", type=Path, default=Path(__file__).resolve().parents[1] / "pipeline", help="Frozen code_active/pipeline directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the frozen front end.")
    parser.add_argument("--tolerance-ms", type=float, default=3.0, help="Maximum nearest-frame time difference (default: 3 ms).")
    parser.add_argument("--max-ref-hz", type=float, default=float("inf"), help="Reference-voiced ceiling; default: no upper cutoff.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Permit fewer than all reference pieces; never use this for the claimed 21-piece result.")
    parser.add_argument("--reuse-frontend-outputs", action="store_true", help="Reuse only already completed generated F0 CSVs in output-dir/frontend_outputs.")
    return parser


def resolve_inputs(pieces: dict[str, Path], args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    names = sorted(pieces)
    overrides: dict[str, dict[str, str]] = {}
    if args.prediction_manifest:
        manifest = read_manifest(args.prediction_manifest.resolve(), "prediction_csv")
        unknown = sorted(set(manifest) - set(names))
        if unknown:
            raise ValueError(f"Prediction manifest contains non-reference pieces: {unknown}")
        paths = {
            piece: resolve_manifest_path(row["prediction_csv"], args.prediction_manifest.resolve())
            for piece, row in manifest.items()
        }
        overrides = manifest
    elif args.prediction_root:
        paths = prediction_candidates(args.prediction_root.resolve(), args.reference_root.resolve(), names)
    else:
        raise ValueError("Provide --prediction-root or --prediction-manifest, unless --run-frozen-frontend is selected.")
    absent = {piece: path for piece, path in paths.items() if not path.is_file()}
    if absent:
        raise FileNotFoundError(f"Prediction files absent: {absent}")
    return paths, overrides


def resolve_audio_inputs(pieces: dict[str, Path], args: argparse.Namespace) -> dict[str, Path]:
    names = sorted(pieces)
    if args.audio_manifest:
        manifest = read_manifest(args.audio_manifest.resolve(), "audio_path")
        unknown = sorted(set(manifest) - set(names))
        if unknown:
            raise ValueError(f"Audio manifest contains non-reference pieces: {unknown}")
        paths = {
            piece: resolve_manifest_path(row["audio_path"], args.audio_manifest.resolve())
            for piece, row in manifest.items()
        }
    elif args.audio_root:
        paths = discover_exact_files(args.audio_root.resolve(), names, WAV_SUFFIXES)
    else:
        raise ValueError("--run-frozen-frontend needs --audio-root or --audio-manifest")
    absent = {piece: path for piece, path in paths.items() if not path.is_file()}
    if absent:
        raise FileNotFoundError(f"Audio files absent: {absent}")
    return paths


def require_complete(reference: dict[str, Path], available: dict[str, Path], label: str, allow_incomplete: bool) -> None:
    missing = sorted(set(reference) - set(available))
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Missing {label} for {len(missing)}/{len(reference)} pieces: {missing}. "
            "Use a complete 21-piece set or explicitly pass --allow-incomplete for debugging only."
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.tolerance_ms <= 0:
        raise ValueError("--tolerance-ms must be positive")
    if args.max_ref_hz <= 0:
        raise ValueError("--max-ref-hz must be positive")
    if args.run_frozen_frontend and (args.prediction_root or args.prediction_manifest):
        raise ValueError("Choose either --run-frozen-frontend or existing predictions, not both.")

    reference_root = args.reference_root.resolve()
    output_dir = args.output_dir.resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root does not exist: {reference_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.reuse_frontend_outputs:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces = reference_pieces(reference_root)
    candidate_piece_count = len(pieces)
    print(f"Found {candidate_piece_count} CCOM-HuQin reference pieces.", flush=True)

    exclusion_rows: list[dict[str, str]] = []
    if args.exclude_manifest:
        exclusion_path = args.exclude_manifest.resolve()
        excluded = read_manifest(exclusion_path, "piece")
        unknown = sorted(set(excluded) - set(pieces))
        if unknown:
            raise ValueError(f"Exclusion manifest contains non-reference pieces: {unknown}")
        exclusion_rows = [
            {"piece": piece, "reason": row.get("reason", "")}
            for piece, row in sorted(excluded.items())
        ]
        pieces = {piece: path for piece, path in pieces.items() if piece not in excluded}
        if not pieces:
            raise ValueError("QC exclusion manifest removes every reference piece")
        print(f"QC exclusions: {len(exclusion_rows)}; evaluating {len(pieces)} pieces.", flush=True)

    if args.run_frozen_frontend:
        audio_paths = resolve_audio_inputs(pieces, args)
        require_complete(pieces, audio_paths, "WAV files", args.allow_incomplete)
        prediction_paths = run_frozen_frontend(audio_paths, output_dir, args.pipeline_dir.resolve(), args)
        overrides: dict[str, dict[str, str]] = {}
    else:
        prediction_paths, overrides = resolve_inputs(pieces, args)
    require_complete(pieces, prediction_paths, "prediction files", args.allow_incomplete)

    rows: list[dict[str, object]] = []
    for piece in sorted(prediction_paths):
        rows.append(score_piece(piece, pieces[piece], prediction_paths[piece], args, overrides.get(piece, {})))
    per_piece = pd.DataFrame(rows).sort_values("piece").reset_index(drop=True)
    if per_piece.empty:
        raise RuntimeError("No pieces were evaluated")

    n_reference = int(per_piece["reference_voiced_frames"].sum())
    n_total = int(per_piece["reference_total_frames"].sum())
    n_covered = int(per_piece["covered_frames"].sum())
    n_correct = int(per_piece["correct_rpa50_frames"].sum())
    n_true_unvoiced = int(per_piece["true_unvoiced_frames"].sum())
    n_joint_correct = int(per_piece["joint_f0_voicing_correct_frames"].sum())
    n_false_alarm = int(per_piece["voicing_false_alarm_frames"].sum())
    n_predicted_voiced = n_covered + n_false_alarm
    summary = pd.DataFrame(
        [
            {
                "dataset": "CCOM-HuQin",
                "candidate_pieces_discovered": candidate_piece_count,
                "pieces_after_qc": len(pieces),
                "pieces_evaluated": len(per_piece),
                "metric": "strict_RPA50_reference_voiced",
                "nearest_time_tolerance_ms": args.tolerance_ms,
                "reference_voiced_max_hz": args.max_ref_hz if np.isfinite(args.max_ref_hz) else "no_upper_cutoff",
                "reference_total_frames": n_total,
                "reference_voiced_frames": n_reference,
                "covered_frames": n_covered,
                "correct_rpa50_frames": n_correct,
                "coverage_pct_macro": float(per_piece["coverage_pct"].mean()),
                "rpa50_pct_macro": float(per_piece["rpa50_pct"].mean()),
                "mutually_voiced_rpa50_pct_macro": float(per_piece["mutually_voiced_rpa50_pct"].mean()),
                "coverage_pct_micro": 100.0 * n_covered / n_reference,
                "rpa50_pct_micro": 100.0 * n_correct / n_reference,
                "mutually_voiced_rpa50_pct_micro": 100.0 * n_correct / n_covered if n_covered else np.nan,
                "reference_unvoiced_frames": int(per_piece["reference_unvoiced_frames"].sum()),
                "true_unvoiced_frames": n_true_unvoiced,
                "voicing_false_alarm_frames": n_false_alarm,
                "voicing_false_alarm_pct_macro": float(per_piece["voicing_false_alarm_pct"].mean()),
                "voicing_false_alarm_pct_micro": 100.0 * n_false_alarm / per_piece["reference_unvoiced_frames"].sum() if int(per_piece["reference_unvoiced_frames"].sum()) else np.nan,
                "voicing_precision_pct_macro": float(per_piece["voicing_precision_pct"].mean()),
                "voicing_precision_pct_micro": 100.0 * n_covered / n_predicted_voiced if n_predicted_voiced else np.nan,
                "joint_f0_voicing_accuracy_pct_macro": float(per_piece["joint_f0_voicing_accuracy_pct"].mean()),
                "joint_f0_voicing_accuracy_pct_micro": 100.0 * n_joint_correct / n_total if n_total else np.nan,
            }
        ]
    )
    per_piece.to_csv(output_dir / "per_piece_rpa50.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary_rpa50.csv", index=False, encoding="utf-8-sig")
    if exclusion_rows:
        pd.DataFrame(exclusion_rows).to_csv(output_dir / "qc_exclusions.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "reference_root": str(reference_root),
        "output_dir": str(output_dir),
        "qc_exclusion_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else None,
        "qc_exclusions": exclusion_rows,
        "run_frozen_frontend": bool(args.run_frozen_frontend),
        "metric_definition": "RPA50: reference-voiced frames only, nearest prediction within tolerance, missing prediction fails, absolute cents error <= 50. Joint F0+voicing: additionally counts a reference-silent frame correct only when prediction is silent.",
        "important_scope": "Expanded 21-piece validation output. It does not overwrite or retroactively alter the formal three-piece frozen result.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + summary.round(4).to_string(index=False))
    print(f"\nSaved: {output_dir / 'per_piece_rpa50.csv'}")
    print(f"Saved: {output_dir / 'summary_rpa50.csv'}")
    print(f"Saved: {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
