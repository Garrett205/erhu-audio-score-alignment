from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch


HERE = Path(__import__("os").environ.get("ERHU_TAPE_WORK", "work/tape" )).resolve()
TAPE_REPO = Path(os.environ["TAPE_REPO"]).resolve()
MANIFEST = HERE / "inference_manifest_18.csv"
EXPECTED_COLUMNS = {"recording_id", "wav_path", "wav_sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    points = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    return {f"{prefix}_p{point}": float(np.percentile(values, point)) for point in points}


def main() -> None:
    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1":
        raise RuntimeError("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD must be 1")
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != EXPECTED_COLUMNS:
            raise RuntimeError(
                "Inference manifest must contain only recording_id,wav_path,wav_sha256; "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 inference inputs, found {len(rows)}")
    if any("reference" in key.lower() for key in (rows[0] if rows else {})):
        raise RuntimeError("Reference leakage in inference manifest")

    sys.path.insert(0, str(TAPE_REPO))
    from pitch_estimator import TAPE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAPE(instrument="violin", hop_length=80).to(device)
    if model.sr != 16000 or model.hop_length != 80 or model.labeling.n_bins != 480:
        raise RuntimeError("Unexpected official TAPE configuration")
    if not np.isclose(model.labeling.min_f0_hz, 164.81377846):
        raise RuntimeError("Unexpected TAPE violin min F0")
    if not np.isclose(model.labeling.granularity_c, 12.5):
        raise RuntimeError("Unexpected TAPE violin bin granularity")

    sanity_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        recording_id = row["recording_id"]
        wav = Path(row["wav_path"])
        if sha256(wav).lower() != row["wav_sha256"].lower():
            raise RuntimeError(f"{recording_id}: WAV SHA-256 mismatch")
        out_dir = HERE / "outputs" / recording_id
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "tape_raw.csv"
        activation_path = out_dir / "tape_activation.npy"
        log_path = out_dir / "run.log"
        existing = (csv_path.exists(), activation_path.exists(), log_path.exists())
        if all(existing):
            saved = json.loads(log_path.read_text(encoding="utf-8"))
            if saved.get("wav_sha256", "").lower() != row["wav_sha256"].lower():
                raise RuntimeError(f"{recording_id}: saved output WAV hash mismatch")
            csv_frame = pd.read_csv(csv_path)
            activation_saved = np.load(activation_path, mmap_mode="r", allow_pickle=False)
            if len(csv_frame) != activation_saved.shape[0] or activation_saved.shape[1] != 480:
                raise RuntimeError(f"{recording_id}: incomplete saved native output")
            sanity_rows.append(saved["sanity"])
            print(f"[{index:02d}/18] reuse completed {recording_id}: {len(csv_frame)} frames", flush=True)
            continue
        if any(existing):
            raise FileExistsError(f"Incomplete existing inference, refusing to overwrite: {out_dir}")

        started = time.perf_counter()
        audio, loaded_sr = librosa.load(wav, sr=model.sr, mono=True)
        if loaded_sr != 16000 or audio.dtype != np.float32 or audio.ndim != 1:
            raise RuntimeError(f"{recording_id}: unexpected audio load result")
        with torch.no_grad():
            times, frequency, confidence, activation = model.predict(
                torch.tensor(audio), viterbi=False, batch_size=128
            )
        times = np.asarray(times, dtype=np.float64)
        frequency = np.asarray(frequency, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        activation = np.asarray(activation, dtype=np.float32)
        if not (len(times) == len(frequency) == len(confidence) == activation.shape[0]):
            raise RuntimeError(f"{recording_id}: native output length mismatch")
        if activation.ndim != 2 or activation.shape[1] != 480:
            raise RuntimeError(f"{recording_id}: activation shape {activation.shape}")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(frequency)):
            raise RuntimeError(f"{recording_id}: non-finite time/frequency")
        # TAPE constructs time with torch.float32 arange, so long tracks have
        # microsecond-scale representation jitter.  Audit against the intended
        # n*hop/sr grid rather than demanding identical adjacent float32 deltas.
        expected_times = np.arange(len(times), dtype=np.float64) * (80.0 / 16000.0)
        grid_max_abs_error_s = float(np.max(np.abs(times - expected_times)))
        if grid_max_abs_error_s > 2e-5:
            raise RuntimeError(
                f"{recording_id}: native time grid differs from n*0.005 by "
                f"{grid_max_abs_error_s:.9f}s"
            )

        pd.DataFrame(
            {"time": times, "frequency": frequency, "confidence": confidence}
        ).to_csv(csv_path, index=False, encoding="utf-8", float_format="%.10g")
        np.save(activation_path, activation, allow_pickle=False)
        elapsed = time.perf_counter() - started
        finite_confidence = confidence[np.isfinite(confidence)]
        sanity = {
            "recording_id": recording_id,
            "wav_sha256": row["wav_sha256"],
            "audio_samples_16k": len(audio),
            "audio_duration_s": len(audio) / 16000.0,
            "output_frames": len(times),
            "time_min_s": float(times.min()),
            "time_max_s": float(times.max()),
            "time_step_median_s": float(np.median(np.diff(times))) if len(times) > 1 else np.nan,
            "time_grid_max_abs_error_s": grid_max_abs_error_s,
            "frequency_min_hz": float(frequency.min()),
            "frequency_max_hz": float(frequency.max()),
            "frequency_median_hz": float(np.median(frequency)),
            "frequency_zero_count": int(np.sum(frequency == 0)),
            "frequency_nonfinite_count": int(np.sum(~np.isfinite(frequency))),
            "confidence_nonfinite_count": int(np.sum(~np.isfinite(confidence))),
            "activation_nonfinite_count": int(np.sum(~np.isfinite(activation))),
            "runtime_seconds": elapsed,
        }
        if finite_confidence.size:
            sanity.update(percentiles(finite_confidence, "confidence"))
        sanity_rows.append(sanity)
        log = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_id": recording_id,
            "wav_path": str(wav),
            "wav_sha256": row["wav_sha256"],
            "python": sys.executable,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "call": "TAPE(instrument='violin', hop_length=80).predict(audio, viterbi=False, batch_size=128)",
            "confidence_used_for_scoring": False,
            "postprocessing": None,
            "outputs": {"csv": str(csv_path), "activation": str(activation_path)},
            "sanity": sanity,
        }
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[{index:02d}/18] {recording_id}: {len(times)} frames, "
            f"F0 {frequency.min():.2f}-{frequency.max():.2f} Hz, {elapsed:.1f}s",
            flush=True,
        )

    pd.DataFrame(sanity_rows).to_csv(
        HERE / "diagnostics" / "tape_output_sanity.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Completed {len(sanity_rows)} independent recordings on {device}")


if __name__ == "__main__":
    main()
