"""Export the original native frame/time mapping for diagnostic readout."""
from pathlib import Path
import os,json
import numpy as np
import pandas as pd

def main():
    root=Path(os.environ.get('ERHU_BASELINE_WORK','work/external'))
    for row in pd.read_csv(root/'00_manifest/benchmark_manifest_12.csv').to_dict('records'):
        cid=row['performance_id']; d=root/'02_alignment/audio_to_score'/cid
        native=json.loads((d/'alignment_native.json').read_text(encoding='utf-8'))
        with np.load(d/'native_acoustic_evidence.npz') as z:path=z['path'].copy()
        sna=np.load(root/'00_manifest/symbolic'/f'{cid}.npy')
        onsets=np.unique(sna['onset_beat'])
        mapped={e['score_id']:e['performance_time'] for e in native}
        by_onset={float(s['onset_beat']):mapped[str(s['id'])] for s in sna}
        ordered_times=np.array([by_onset[float(s)] for s in onsets])
        shift=ordered_times-np.asarray(path).reshape(-1)/50
        assert np.ptp(shift)<1e-10
        dest=root/'09_readonly_audit/audio_native'/cid;dest.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(score_position=onsets,raw_frame_index=path,raw_predicted_time=ordered_times)).to_csv(dest/'native_positions_frames_times.csv',index=False)

if __name__=='__main__':main()
