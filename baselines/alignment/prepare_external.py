"""Build baseline input arrays from authorized corrected scores; no score editing."""
from pathlib import Path
import argparse,os,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/pipeline'))
import mxml as m
import pitch_dtw

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path(os.environ.get('ERHU_BASELINE_WORK','work/external')))
    p.add_argument('--runs',type=Path,default=Path(os.environ.get('ERHU_RUNS','work/runs')))
    args=p.parse_args(); out=args.work
    manifest=pd.read_csv(out/'00_manifest/benchmark_manifest_12.csv')
    all_events=[]
    for row in manifest.to_dict('records'):
        cid=row['performance_id']
        f=pitch_dtw.load_score(args.runs/cid/f'{cid}_score_final.csv')
        score=m.m21.converter.parse(row['musicxml_path'],forceSource=True)
        raw,_,_=m.make_note_rows(score,m.build_tempo_map(score),m.collect_wedge_intervals(Path(row['musicxml_path']),score))
        selected=raw.set_index('score_note_id').loc[f.score_note_id].reset_index()
        assert selected.score_note_id.tolist()==f.score_note_id.tolist()
        assert np.allclose(selected.pitch_midi,69+12*np.log2(f.score_pitch_hz/440),atol=.001)
        assert np.allclose(selected.onset_quarter,f['谱面起点(拍)'],atol=.00001)
        a=np.zeros(len(f),dtype=[('id','U128'),('pitch','i4'),('onset_beat','f8'),('duration_beat','f8'),('onset_quarter','f8'),('duration_quarter','f8'),('onset_div','i4'),('duration_div','i4'),('is_grace','?'),('voice','i4')])
        a['id']=selected.score_note_id
        a['pitch']=selected.pitch_midi
        a['onset_beat']=selected.onset_quarter
        a['onset_quarter']=selected.onset_quarter
        a['duration_beat']=pd.to_numeric(f['逻辑持续时间(拍)'])
        a['duration_quarter']=a['duration_beat']
        a['onset_div']=np.rint(a['onset_beat']*480).astype(int)
        a['duration_div']=np.rint(a['duration_beat']*480).astype(int)
        a['is_grace']=selected.is_grace
        a['voice']=1
        dest=out/'00_manifest/symbolic';dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/f'{cid}.npy',a)
        f.insert(0,'performance_id',cid);all_events.append(f)
    pd.concat(all_events).to_csv(out/'00_manifest/frozen_score_events.csv',index=False)

if __name__=='__main__':main()
