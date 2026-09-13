from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__import__("os").environ.get("ERHU_PYIN_WORK", "work/pyin" )).resolve()
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = HERE / "benchmark_manifest_18.csv"
FROZEN_EVALUATOR = ROOT / "src/evaluation/pitch_metrics.py"
EXPECTED = (18, 198_536, 164_183, 34_353)
NOTE_MIDI = list(range(62, 95))
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("frozen_pitch_evaluator_pyin", FROZEN_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(FROZEN_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def note_name(midi: int) -> str:
    return f"{PITCH_CLASSES[midi % 12]}{midi // 12 - 1}"


def main() -> None:
    evaluator = load_evaluator()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    args = argparse.Namespace(tolerance_ms=3.0, max_ref_hz=float("inf"))
    per_rows = []
    note_arrays = []
    for item in manifest:
        recording_id = item["recording_id"]
        reference_path = Path(item["reference_pitch_path"])
        prediction_path = HERE / "outputs" / recording_id / "pyin_raw.csv"
        core = evaluator.score_piece(recording_id, reference_path, prediction_path, args, {})
        reference = evaluator.load_track(reference_path)
        prediction = evaluator.load_track(prediction_path)
        estimate = evaluator.nearest_prediction(reference.time_s, prediction.time_s, prediction.f0_hz, 0.003)
        signed = evaluator.cents_error(estimate, reference.f0_hz)
        voiced = np.isfinite(reference.f0_hz) & (reference.f0_hz > 0)
        unvoiced = np.isfinite(reference.f0_hz) & (reference.f0_hz <= 0)
        covered = voiced & np.isfinite(estimate) & (estimate > 0)
        up = covered & (np.abs(signed - 1200.0) <= 50.0)
        down = covered & (np.abs(signed + 1200.0) <= 50.0)
        hf = (
            np.isfinite(estimate)
            & (estimate >= 1900.0)
            & (estimate <= 2000.0)
            & (unvoiced | (voiced & (np.abs(signed) >= 100.0)))
        )
        n_voiced = int(voiced.sum())
        n_total = int((voiced | unvoiced).sum())
        per_rows.append(
            {
                **core,
                "overall50_numerator": int(core["joint_f0_voicing_correct_frames"]),
                "overall50_denominator": n_total,
                "overall50_pct": float(core["joint_f0_voicing_accuracy_pct"]),
                "voiced_rpa50_numerator": int(core["correct_rpa50_frames"]),
                "voiced_rpa50_denominator": n_voiced,
                "voiced_rpa50_pct": float(core["rpa50_pct"]),
                "coverage_numerator": int(core["covered_frames"]),
                "coverage_denominator": n_voiced,
                "octave_up_numerator": int(up.sum()),
                "octave_up_denominator": n_voiced,
                "octave_up_pct": pct(int(up.sum()), n_voiced),
                "octave_down_numerator": int(down.sum()),
                "octave_down_denominator": n_voiced,
                "octave_down_pct": pct(int(down.sum()), n_voiced),
                "failure_1900_2000_numerator": int(hf.sum()),
                "failure_1900_2000_denominator": n_total,
                "failure_1900_2000_pct": pct(int(hf.sum()), n_total),
            }
        )
        note_arrays.append((reference.f0_hz, signed))
    per = pd.DataFrame(per_rows).sort_values("piece").reset_index(drop=True)
    totals = (
        len(per),
        int(per["reference_total_frames"].sum()),
        int(per["reference_voiced_frames"].sum()),
        int(per["reference_unvoiced_frames"].sum()),
    )
    if totals != EXPECTED:
        raise RuntimeError(f"Frozen evaluation gate failed: {totals}")
    per.to_csv(HERE / "evaluation" / "pyin_per_recording_metrics.csv", index=False, encoding="utf-8-sig")
    counts = {
        "overall50_numerator": int(per["overall50_numerator"].sum()),
        "overall50_denominator": EXPECTED[1],
        "voiced_rpa50_numerator": int(per["voiced_rpa50_numerator"].sum()),
        "voiced_rpa50_denominator": EXPECTED[2],
        "coverage_numerator": int(per["coverage_numerator"].sum()),
        "coverage_denominator": EXPECTED[2],
        "octave_up_numerator": int(per["octave_up_numerator"].sum()),
        "octave_up_denominator": EXPECTED[2],
        "octave_down_numerator": int(per["octave_down_numerator"].sum()),
        "octave_down_denominator": EXPECTED[2],
        "failure_1900_2000_numerator": int(per["failure_1900_2000_numerator"].sum()),
        "failure_1900_2000_denominator": EXPECTED[1],
        "reference_unvoiced_frames": EXPECTED[3],
        "true_unvoiced_frames": int(per["true_unvoiced_frames"].sum()),
        "voicing_false_alarm_frames": int(per["voicing_false_alarm_frames"].sum()),
    }
    pooled = {
        "method": "Raw pYIN (librosa)",
        "recordings": EXPECTED[0],
        "reference_total_frames": EXPECTED[1],
        "reference_voiced_frames": EXPECTED[2],
        "reference_unvoiced_frames": EXPECTED[3],
        **counts,
        "overall50_pct": pct(counts["overall50_numerator"], EXPECTED[1]),
        "voiced_rpa50_pct": pct(counts["voiced_rpa50_numerator"], EXPECTED[2]),
        "coverage_pct": pct(counts["coverage_numerator"], EXPECTED[2]),
        "octave_up_pct": pct(counts["octave_up_numerator"], EXPECTED[2]),
        "octave_down_pct": pct(counts["octave_down_numerator"], EXPECTED[2]),
        "failure_1900_2000_pct": pct(counts["failure_1900_2000_numerator"], EXPECTED[1]),
        "macro_overall50_pct": float(per["overall50_pct"].mean()),
        "macro_voiced_rpa50_pct": float(per["voiced_rpa50_pct"].mean()),
        "macro_coverage_pct": float(per["coverage_pct"].mean()),
        "macro_octave_up_pct": float(per["octave_up_pct"].mean()),
        "macro_octave_down_pct": float(per["octave_down_pct"].mean()),
        "macro_failure_1900_2000_pct": float(per["failure_1900_2000_pct"].mean()),
    }
    pd.DataFrame([pooled]).to_csv(HERE / "evaluation" / "pyin_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    (HERE / "evaluation" / "pyin_confusion_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{
            "Method": pooled["method"],
            "Overall-50": pooled["overall50_pct"],
            "voiced RPA50": pooled["voiced_rpa50_pct"],
            "coverage": pooled["coverage_pct"],
            "octave-up": pooled["octave_up_pct"],
            "octave-down": pooled["octave_down_pct"],
            "1900-2000 failure": pooled["failure_1900_2000_pct"],
        }]
    ).to_csv(HERE / "evaluation" / "table3_candidate_row.csv", index=False, encoding="utf-8-sig")

    all_ref = np.concatenate([item[0] for item in note_arrays])
    all_err = np.concatenate([item[1] for item in note_arrays])
    ref_midi = np.full(len(all_ref), np.nan)
    positive = np.isfinite(all_ref) & (all_ref > 0)
    ref_midi[positive] = 69.0 + 12.0 * np.log2(all_ref[positive] / 440.0)
    assigned = np.full(len(all_ref), -1, dtype=np.int16)
    in_range = positive & (ref_midi >= 61.5) & (ref_midi <= 94.5)
    assigned[in_range] = np.clip(np.floor(ref_midi[in_range] + 0.5), 62, 94).astype(np.int16)
    note_rows = []
    for midi in NOTE_MIDI:
        mask = assigned == midi
        n = int(mask.sum())
        correct = int((mask & np.isfinite(all_err) & (np.abs(all_err) <= 50.0)).sum())
        up = int((mask & np.isfinite(all_err) & (np.abs(all_err - 1200.0) <= 50.0)).sum())
        down = int((mask & np.isfinite(all_err) & (np.abs(all_err + 1200.0) <= 50.0)).sum())
        note_rows.append(
            {
                "note_midi": midi,
                "note_name": note_name(midi),
                "lower_bound_cents_from_note": -50,
                "upper_bound_cents_from_note": 50,
                "N": n,
                "rpa50_numerator": correct,
                "RPA50_pct": pct(correct, n),
                "octave_up_numerator": up,
                "octave_up_pct": pct(up, n),
                "octave_down_numerator": down,
                "octave_down_pct": pct(down, n),
            }
        )
    pd.DataFrame(note_rows).to_csv(
        HERE / "diagnostics" / "pyin_note_wise_D4_Asharp6.csv", index=False, encoding="utf-8-sig"
    )
    config_path = HERE / "freeze" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["executed_evaluation_python"] = sys.executable
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(pooled, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
