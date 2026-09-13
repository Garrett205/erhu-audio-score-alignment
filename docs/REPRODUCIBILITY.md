# Reproduction scope and entry points

## Proposed pipeline

`python run_pipeline.py --input inputs/performance.wav --score inputs/score.musicxml --output work/example`
copies inputs to a new directory and runs PMSDB → pitch → mxml → onset → os →
DP1 → DP2 → DP3 → final unmatched-only Mel selection. Each original stage uses
`ERHU_WORK_DIR`. Run individual scripts with that variable to inspect an existing
work directory. Do not put user inputs inside the source directory.

`configs/frozen_constants.json` records literal module constants;
`configs/paper_config.json` indexes the paper policy. Parameters remain defined
in the original modules; these JSON files do not silently override them.

The launcher and generic Mel I/O driver are new packaging adapters, not the
historical twelve-case experimental driver. The selector and helper functions
are copied verbatim. The final paper invocation used best-admissible selection
(zero runner-up margin for Mel only); ordinary DP3 ambiguity rules remain intact.
No-anchor/no-positive-peak inputs preserve the unresolved event. Processing
multiple unresolved events through this adapter is outside the demonstrated
two-development-event Mel evidence; no additional accuracy claim is made.

Physical candidate thresholds use the original 5-ms audio axis. Global
normalization changes the performance axis before DP; DP spacing, timing costs
and recovery bounds then use the normalized axis. External baseline readout
stays on physical time. Gate output is a candidate-domain boundary, never a
physical release annotation or audio trim. DP1 assumes at least as many strong
candidates as active score events, as in the evaluated performances.

## Public pitch benchmark

`python src/evaluation/pitch_metrics.py --help` lists input/reference manifests,
prediction roots and frontend-only execution options. Its default nearest-time
tolerance is 3 ms. Restrict to the paper's authorized 18-recording population;
the code supports a QC exclusion manifest but no dataset records are bundled.
pYIN/TAPE commands and manifest schemas are in `baselines/README.md`.
Publicly obtained CCOM data support recomputing these pitch metrics, but exact
paper totals require the same input versions and retained population.

## Private alignment benchmark

The proposed method and baseline wrapper code are provided. Reproduction of
the paper's private alignment numbers requires recordings, corrected scores,
and audit evidence that this repository does not distribute. The public CCOM
recordings are not interchangeable with the separate private performances.
Manual acceptance and automatic pitch-valid baseline output are different
endpoints; neither is relabeled as the other.

## Extraction and validation

`SOURCE_PROVENANCE.json` records source/output SHA-256 and unchanged/changed
function ASTs. The entire pitch, onset, DP1/2/3 algorithm files are unchanged.
Only irrelevant score technique exports, unused legacy score merging and
downstream true-stage wrappers were removed. Score IDs, order, ties and grace
semantics are retained. Some historical stage-local diagnostic labels remain
because they are coupled to the actual output/validity processing.

`python verify_snapshot.py` compiles source in memory, verifies its hash manifest,
and checks forbidden artifacts and gate constants without data or inference.
`python tests/smoke.py` tests synthetic gate, pitch repair, score identity and
baseline readout behavior. No model inference, private-data alignment rerun,
new manual audit or clean-environment rebuild was performed.
