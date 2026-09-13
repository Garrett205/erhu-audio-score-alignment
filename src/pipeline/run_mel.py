"""Apply the frozen unmatched-only Mel selector to explicit work inputs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mel_rescue as rescue
import mel_support as support
from mel_onset import extract
from score_metadata_bridge import load_full_and_alignment_score


def one(root, pattern):
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f'Expected one {pattern}; found {len(paths)}')
    return paths[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    work = args.work.resolve()
    src = one(work, '*_DP3_对齐结果_score主导.csv')
    base = pd.read_csv(src)
    on = pd.read_csv(one(work, '*_onset_scaled.csv'))
    pitch = pd.read_csv(one(work, '*_pitch2.csv'))
    _, _, score = load_full_and_alignment_score(work)
    dest = work / 'final'
    dest.mkdir(exist_ok=False)
    indices = np.flatnonzero(~support.valid(base))
    final = base.copy()
    changed = []
    if len(indices):
        curve, _ = extract(args.input)
        curve.to_csv(dest / 'mel_novelty.csv', index=False)
        for i in indices:
            left, right = rescue.dp._nearest_recovery_anchors(final, int(i))
            if left is None or right is None:
                continue  # The frozen selector requires two valid boundaries.
            lo, hi = float(final.loc[left, support.T]), float(final.loc[right, support.T])
            frames = np.union1d(rescue.find_peaks(support.nums(curve, 'novelty_raw'))[0],
                                rescue.find_peaks(support.nums(curve, 'novelty_norm'))[0])
            if not len(frames):
                continue
            times = support.scaled(support.nums(curve, 'raw_time_s')[frames], on)
            positive = ((support.nums(curve, 'novelty_raw')[frames] > 0) |
                        (support.nums(curve, 'novelty_norm')[frames] > 0))
            if not np.any((times > lo) & (times < hi) & positive):
                continue
            result, diag, _, _, _ = rescue.candidates('input', final, pitch, on, curve, score, int(i))
            diag.to_csv(dest / f'mel_candidates_{i}.csv', index=False)
            if rescue.dp._valid_matched_row(result.loc[i]):
                final.loc[i] = result.loc[i, base.columns]
                changed.append(int(i))
    rescue.write_exact_patch(src, dest / src.name, final, changed)
    pd.testing.assert_frame_equal(base.loc[support.valid(base)], final.loc[support.valid(base)],
                                  check_dtype=False, check_exact=True)


if __name__ == '__main__':
    main()
