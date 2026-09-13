from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__import__("os").environ.get("ERHU_TAPE_WORK", "work/tape" )).resolve()
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = HERE / "benchmark_manifest_18.csv"
FROZEN_EVALUATOR = ROOT / "src/evaluation/pitch_metrics.py"
EXPECTED = (18, 198_536, 164_183, 34_353)
TOLERANCE_MS = 3.0
NOTE_MIDI = list(range(62, 95))  # D4 through A#6 inclusive
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    return f"{PITCH_CLASSES[midi % 12]}{midi // 12 - 1}"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("frozen_pitch_evaluator", FROZEN_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FROZEN_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def main() -> None:
    evaluator = load_evaluator()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != EXPECTED[0]:
        raise RuntimeError(f"Manifest row count {len(manifest)}")
    args = argparse.Namespace(tolerance_ms=TOLERANCE_MS, max_ref_hz=float("inf"))
    per_rows: list[dict[str, object]] = []
    note_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_count = all_voiced = all_unvoiced = 0

    for row in manifest:
        recording_id = row["recording_id"]
        reference_path = Path(row["reference_pitch_path"])
        prediction_path = HERE / "outputs" / recording_id / "tape_raw.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)

        core = evaluator.score_piece(recording_id, reference_path, prediction_path, args, {})
        reference = evaluator.load_track(reference_path)
        prediction = evaluator.load_track(prediction_path)
        estimate = evaluator.nearest_prediction(
            reference.time_s, prediction.time_s, prediction.f0_hz, TOLERANCE_MS / 1000.0
        )
        signed_error = evaluator.cents_error(estimate, reference.f0_hz)
        valid_ref = np.isfinite(reference.f0_hz)
        voiced = valid_ref & (reference.f0_hz > 0)
        unvoiced = valid_ref & (reference.f0_hz <= 0)
        covered = voiced & np.isfinite(estimate) & (estimate > 0)
        octave_up = covered & (np.abs(signed_error - 1200.0) <= 50.0)
        octave_down = covered & (np.abs(signed_error + 1200.0) <= 50.0)
        ceiling = (
            np.isfinite(estimate)
            & (estimate >= 1900.0)
            & (estimate <= 2000.0)
            & (unvoiced | (voiced & (np.abs(signed_error) >= 100.0)))
        )
        total = int(np.sum(voiced | unvoiced))
        n_voiced = int(np.sum(voiced))
        n_unvoiced = int(np.sum(unvoiced))
        all_count += total
        all_voiced += n_voiced
        all_unvoiced += n_unvoiced
        per_rows.append(
            {
                **core,
                "overall50_numerator": int(core["joint_f0_voicing_correct_frames"]),
                "overall50_denominator": total,
                "overall50_pct": float(core["joint_f0_voicing_accuracy_pct"]),
                "voiced_rpa50_numerator": int(core["correct_rpa50_frames"]),
                "voiced_rpa50_denominator": n_voiced,
                "voiced_rpa50_pct": float(core["rpa50_pct"]),
                "coverage_numerator": int(core["covered_frames"]),
                "coverage_denominator": n_voiced,
                "octave_up_numerator": int(np.sum(octave_up)),
                "octave_up_denominator": n_voiced,
                "octave_up_pct": safe_pct(int(np.sum(octave_up)), n_voiced),
                "octave_down_numerator": int(np.sum(octave_down)),
                "octave_down_denominator": n_voiced,
                "octave_down_pct": safe_pct(int(np.sum(octave_down)), n_voiced),
                "failure_1900_2000_numerator": int(np.sum(ceiling)),
                "failure_1900_2000_denominator": total,
                "failure_1900_2000_pct": safe_pct(int(np.sum(ceiling)), total),
            }
        )
        note_arrays.append((reference.f0_hz, estimate, signed_error))

    if (len(per_rows), all_count, all_voiced, all_unvoiced) != EXPECTED:
        raise RuntimeError(
            f"Evaluation denominator mismatch: {(len(per_rows), all_count, all_voiced, all_unvoiced)}"
        )
    per = pd.DataFrame(per_rows).sort_values("piece").reset_index(drop=True)
    per.to_csv(
        HERE / "evaluation" / "tape_per_recording_metrics.csv", index=False, encoding="utf-8-sig"
    )

    pooled_counts = {
        "overall50_numerator": int(per["overall50_numerator"].sum()),
        "overall50_denominator": all_count,
        "voiced_rpa50_numerator": int(per["voiced_rpa50_numerator"].sum()),
        "voiced_rpa50_denominator": all_voiced,
        "coverage_numerator": int(per["coverage_numerator"].sum()),
        "coverage_denominator": all_voiced,
        "octave_up_numerator": int(per["octave_up_numerator"].sum()),
        "octave_up_denominator": all_voiced,
        "octave_down_numerator": int(per["octave_down_numerator"].sum()),
        "octave_down_denominator": all_voiced,
        "failure_1900_2000_numerator": int(per["failure_1900_2000_numerator"].sum()),
        "failure_1900_2000_denominator": all_count,
        "reference_unvoiced_frames": all_unvoiced,
        "true_unvoiced_frames": int(per["true_unvoiced_frames"].sum()),
        "voicing_false_alarm_frames": int(per["voicing_false_alarm_frames"].sum()),
    }
    pooled = {
        "method": "Raw TAPE-violin",
        "recordings": len(per),
        "reference_total_frames": all_count,
        "reference_voiced_frames": all_voiced,
        "reference_unvoiced_frames": all_unvoiced,
        **pooled_counts,
        "overall50_pct": safe_pct(pooled_counts["overall50_numerator"], all_count),
        "voiced_rpa50_pct": safe_pct(pooled_counts["voiced_rpa50_numerator"], all_voiced),
        "coverage_pct": safe_pct(pooled_counts["coverage_numerator"], all_voiced),
        "octave_up_pct": safe_pct(pooled_counts["octave_up_numerator"], all_voiced),
        "octave_down_pct": safe_pct(pooled_counts["octave_down_numerator"], all_voiced),
        "failure_1900_2000_pct": safe_pct(pooled_counts["failure_1900_2000_numerator"], all_count),
        "macro_overall50_pct": float(per["overall50_pct"].mean()),
        "macro_voiced_rpa50_pct": float(per["voiced_rpa50_pct"].mean()),
        "macro_coverage_pct": float(per["coverage_pct"].mean()),
        "macro_octave_up_pct": float(per["octave_up_pct"].mean()),
        "macro_octave_down_pct": float(per["octave_down_pct"].mean()),
        "macro_failure_1900_2000_pct": float(per["failure_1900_2000_pct"].mean()),
    }
    pd.DataFrame([pooled]).to_csv(
        HERE / "evaluation" / "tape_pooled_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (HERE / "evaluation" / "tape_confusion_counts.json").write_text(
        json.dumps(pooled_counts, indent=2) + "\n", encoding="utf-8"
    )
    candidate_fields = {
        "Method": "Raw TAPE-violin",
        "Overall-50": pooled["overall50_pct"],
        "voiced RPA50": pooled["voiced_rpa50_pct"],
        "coverage": pooled["coverage_pct"],
        "octave-up": pooled["octave_up_pct"],
        "octave-down": pooled["octave_down_pct"],
        "1900-2000 failure": pooled["failure_1900_2000_pct"],
    }
    pd.DataFrame([candidate_fields]).to_csv(
        HERE / "evaluation" / "table3_candidate_row.csv", index=False, encoding="utf-8-sig"
    )

    all_ref = np.concatenate([item[0] for item in note_arrays])
    all_est = np.concatenate([item[1] for item in note_arrays])
    all_err = np.concatenate([item[2] for item in note_arrays])
    ref_midi = np.full(len(all_ref), np.nan)
    positive_ref = np.isfinite(all_ref) & (all_ref > 0)
    ref_midi[positive_ref] = 69.0 + 12.0 * np.log2(all_ref[positive_ref] / 440.0)
    assigned = np.full(len(all_ref), -1, dtype=np.int16)
    in_range = positive_ref & (ref_midi >= 61.5) & (ref_midi <= 94.5)
    assigned[in_range] = np.clip(np.floor(ref_midi[in_range] + 0.5), 62, 94).astype(np.int16)
    note_rows = []
    for midi in NOTE_MIDI:
        mask = assigned == midi
        n = int(np.sum(mask))
        correct = int(np.sum(mask & np.isfinite(all_err) & (np.abs(all_err) <= 50.0)))
        octave_up = int(np.sum(mask & np.isfinite(all_err) & (np.abs(all_err - 1200.0) <= 50.0)))
        octave_down = int(np.sum(mask & np.isfinite(all_err) & (np.abs(all_err + 1200.0) <= 50.0)))
        note_rows.append(
            {
                "note_midi": midi,
                "note_name": note_name(midi),
                "lower_bound_cents_from_note": -50,
                "upper_bound_cents_from_note": 50,
                "N": n,
                "rpa50_numerator": correct,
                "RPA50_pct": safe_pct(correct, n),
                "octave_up_numerator": octave_up,
                "octave_up_pct": safe_pct(octave_up, n),
                "octave_down_numerator": octave_down,
                "octave_down_pct": safe_pct(octave_down, n),
            }
        )
    pd.DataFrame(note_rows).to_csv(
        HERE / "diagnostics" / "tape_note_wise_D4_Asharp6.csv",
        index=False,
        encoding="utf-8-sig",
    )
    run_config_path = HERE / "freeze" / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    inference_pythons = sorted(
        {
            json.loads((HERE / "outputs" / row["recording_id"] / "run.log").read_text(encoding="utf-8"))["python"]
            for row in manifest
        }
    )
    run_config["executed_inference_python"] = inference_pythons
    run_config["executed_evaluation_python"] = sys.executable
    run_config["figures_generated"] = False
    run_config["paper_files_modified"] = False
    run_config_path.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(pooled, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
