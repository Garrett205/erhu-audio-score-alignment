# Frozen baselines

Third-party implementation source/checkpoints are not included. Obtain them from
their original repositories and preserve their licenses. No current upstream code
was substituted for the frozen experimental versions.

| Baseline | Origin/version | Checkpoint |
|---|---|---|
| pYIN | librosa 0.11.0 | none |
| TAPE | https://github.com/MTG/tape at `2c1673b70a6ade53cd75383e717fba2d424ab66a` | `violin_model.pt`, SHA-256 `af82292f62fde2b5168c584290756def8a1b5e557d3c8e1f45b504a8c0692314` |
| STRAdi | https://github.com/seingreen/STRAdi at `6b9612056b2a31775ef833dc38337ed6040b1f8c` | `violin_transcription_offline.pth`, SHA-256 `2b344f91fe5d49499faf553b982ab5944ee6b9be46de07879ca727fa54f64123` |
| TheGlueNote | Parangonar 3.3.3; https://github.com/sildater/parangonar | packaged `thegluenote_small_checkpoint.pt`, SHA-256 `ffdaac52f0730576e9fe45ee18634e92534987dfe17b217eda4369ec6ea0add9` |
| DualDTW, AudioToScoreMatcher | Parangonar 3.3.3 | package implementation; no separate checkpoint recorded |

Optional package versions: miditok 3.0.6.post1, symusic 0.6.0, dtaidistance 2.4.0.
Install each external checkout according to its upstream instructions at the
recorded commit. STRAdi's historical compatibility changes were
`torch.load(..., weights_only=False)` and torchlibrosa 0.0.4 `pad_center` keyword
compatibility with librosa 0.11.0; they are not silently applied by this snapshot.
TAPE requires its original AGPL license. Neither neural package is vendored here.

## pYIN and TAPE

Working roots default to `work/pyin` and `work/tape`; override with
`ERHU_PYIN_WORK` and `ERHU_TAPE_WORK`. Create `inference_manifest_18.csv` there,
with exactly `recording_id,wav_path,wav_sha256` columns and 18 rows. Resolve
relative WAV paths from your shell working directory. For evaluation also create
`benchmark_manifest_18.csv` with `recording_id,reference_pitch_path` columns.
The frozen inference/evaluator scripts intentionally keep the population checks.
Set `TAPE_REPO` to a separately obtained checkout and obtain the violin weights.

```sh
python baselines/pitch/pyin/run_pyin_inference.py
python baselines/pitch/pyin/evaluate_pyin.py
# Set TAPE_REPO and TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in your shell first.
python baselines/pitch/tape/run_tape_inference.py
python baselines/pitch/tape/evaluate_tape.py
```

## DTW

`alignment/pitch_dtw.py:run_piece(score_csv,pitch_csv,piece_id,cfg)` is the
actual pitch-only baseline core (DTWConfig defaults preserved).
`alignment/pitch_onset_dtw.py:run_piece(score_csv,pitch_csv,piece_id,onset_csv,audio_path,cfg)`
is the frozen pitch+onset baseline. Supply the raw CREPE output CSV and PMSDB
onset-strength CSV; do not pass globally scaled DP candidate times.
Use `python baselines/alignment/run_dtw.py --help` for a portable invocation.

## External alignment and readout

Set `ERHU_BASELINE_WORK` (default `work/external`), `ERHU_RUNS` (default
`work/runs`), and `STRADI_REPO` for the separately installed STRAdi checkout.
Prepare `00_manifest/benchmark_manifest_12.csv` with at least
`performance_id,wav_path,musicxml_path,event_count` and supply
`00_manifest/frozen_score_events.csv` with loader-produced score events.
Run `prepare_external.py --help` to generate symbolic arrays from your corrected
authorized scores. No score correction is made automatically.

```sh
python baselines/alignment/prepare_external.py
python baselines/alignment/stradi_once.py
python baselines/alignment/run_alignment.py theglue
python baselines/alignment/run_alignment.py dual_dtw
python baselines/alignment/run_alignment.py audio_to_score
python baselines/alignment/prepare_native_positions.py
python baselines/alignment/audio_to_score_readout.py --output-dir work/external/readout
```

`anchor_readout.py:evaluate(cid,method,150,data)` preserves the actual STRAdi
event-local readout; `data` is `(alignment_df, physical_pitch_times, f0_hz,
valid_mask, next_matched_onset_by_performance_note_id, input_hashes)`.
Its dataframe requires `score_f0_hz` joined by stable score event ID.
`python baselines/alignment/run_readout.py` prepares those inputs and runs the frozen
150-ms window. Native deletions fail. The earliest two adjacent valid 5-ms F0
frames within 100 cents must lie inside the bounded matched-note interval.
AudioToScore readout is a unique-position pitch-valid diagnostic, not manual
correspondence accuracy. Its original 12/1,229 population checks are retained.
