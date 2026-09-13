from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


HERE = Path(__import__("os").environ.get("ERHU_PYIN_WORK", "work/pyin" )).resolve()
MANIFEST = HERE / "inference_manifest_18.csv"
EXPECTED_COLUMNS = {"recording_id", "wav_path", "wav_sha256"}
SR = 16000
HOP = 80
FMIN = float(librosa.note_to_hz("C2"))
FMAX = float(librosa.note_to_hz("C7"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    points = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    return {
        f"{prefix}_p{point}": float(np.percentile(finite, point)) if finite.size else np.nan
        for point in points
    }


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != EXPECTED_COLUMNS:
            raise RuntimeError(f"Inference manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 18 or any("reference" in name.lower() for name in (reader.fieldnames or [])):
        raise RuntimeError("Inference manifest is not the isolated frozen 18-recording input")

    sanity_rows = []
    for index, row in enumerate(rows, start=1):
        recording_id = row["recording_id"]
        wav = Path(row["wav_path"])
        if sha256(wav).lower() != row["wav_sha256"].lower():
            raise RuntimeError(f"{recording_id}: WAV SHA-256 mismatch")
        out_dir = HERE / "outputs" / recording_id
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "pyin_raw.csv"
        log_path = out_dir / "run.log"
        if csv_path.exists() or log_path.exists():
            raise FileExistsError(f"Refusing to overwrite {out_dir}")

        started = time.perf_counter()
        audio, loaded_sr = librosa.load(wav, sr=SR, mono=True)
        if loaded_sr != SR or audio.dtype != np.float32 or audio.ndim != 1:
            raise RuntimeError(f"{recording_id}: unexpected audio load")
        f0, voiced_flag, voiced_probability = librosa.pyin(
            audio,
            fmin=FMIN,
            fmax=FMAX,
            sr=SR,
            frame_length=2048,
            hop_length=HOP,
            n_thresholds=100,
            beta_parameters=(2, 18),
            boltzmann_parameter=2.0,
            resolution=0.1,
            max_transition_rate=35.92,
            switch_prob=0.01,
            no_trough_prob=0.01,
            fill_na=np.nan,
            center=True,
            pad_mode="constant",
        )
        f0 = np.asarray(f0, dtype=np.float64)
        voiced_flag = np.asarray(voiced_flag, dtype=bool)
        voiced_probability = np.asarray(voiced_probability, dtype=np.float64)
        times = np.arange(len(f0), dtype=np.float64) * HOP / SR
        if not (len(f0) == len(voiced_flag) == len(voiced_probability)):
            raise RuntimeError(f"{recording_id}: native output length mismatch")
        finite_f0 = np.isfinite(f0)
        if not np.array_equal(finite_f0, voiced_flag):
            raise RuntimeError(f"{recording_id}: native F0/voiced_flag mismatch")
        if np.any(f0[finite_f0] < FMIN - 1e-9) or np.any(f0[finite_f0] > FMAX + 1e-9):
            raise RuntimeError(f"{recording_id}: finite F0 outside frozen pYIN bounds")
        if np.any(~np.isfinite(voiced_probability)):
            raise RuntimeError(f"{recording_id}: non-finite native voiced probability")

        pd.DataFrame(
            {
                "time": times,
                "frequency": f0,
                "voiced_flag": voiced_flag,
                "voiced_probability": voiced_probability,
            }
        ).to_csv(csv_path, index=False, encoding="utf-8", float_format="%.10g", na_rep="NaN")
        elapsed = time.perf_counter() - started
        sanity = {
            "recording_id": recording_id,
            "wav_sha256": row["wav_sha256"],
            "audio_samples_16k": len(audio),
            "audio_duration_s": len(audio) / SR,
            "output_frames": len(f0),
            "time_min_s": float(times.min()),
            "time_max_s": float(times.max()),
            "time_step_median_s": float(np.median(np.diff(times))),
            "finite_f0_frames": int(finite_f0.sum()),
            "native_unvoiced_frames": int((~voiced_flag).sum()),
            "frequency_min_hz": float(np.min(f0[finite_f0])) if finite_f0.any() else np.nan,
            "frequency_max_hz": float(np.max(f0[finite_f0])) if finite_f0.any() else np.nan,
            "frequency_median_hz": float(np.median(f0[finite_f0])) if finite_f0.any() else np.nan,
            "f0_nan_count": int(np.isnan(f0).sum()),
            "f0_infinite_count": int(np.isinf(f0).sum()),
            "voiced_probability_nonfinite_count": int((~np.isfinite(voiced_probability)).sum()),
            "runtime_seconds": elapsed,
            **finite_percentiles(voiced_probability, "voiced_probability"),
        }
        sanity_rows.append(sanity)
        log = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_id": recording_id,
            "wav_path": str(wav),
            "wav_sha256": row["wav_sha256"],
            "python": sys.executable,
            "librosa": librosa.__version__,
            "call": "librosa.pyin with explicitly frozen librosa-0.11 defaults, sr=16000, hop_length=80, fmin=C2, fmax=C7",
            "additional_probability_threshold": None,
            "postprocessing": None,
            "native_nan_preserved_in_csv": True,
            "sanity": sanity,
        }
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[{index:02d}/18] {recording_id}: {len(f0)} frames, "
            f"voiced={finite_f0.sum()}, {elapsed:.1f}s",
            flush=True,
        )
    pd.DataFrame(sanity_rows).to_csv(
        HERE / "diagnostics" / "pyin_output_sanity.csv", index=False, encoding="utf-8-sig"
    )
    config_path = HERE / "freeze" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "librosa_version": librosa.__version__,
            "fmin_hz": FMIN,
            "fmax_hz": FMAX,
            "executed_inference_python": sys.executable,
        }
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
