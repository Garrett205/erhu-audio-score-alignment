"""Read original matcher outputs with the paper's fixed 150-ms window."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from anchor_readout import evaluate,summary

def main():
    root=Path(os.environ.get('ERHU_BASELINE_WORK','work/external'))
    runs=Path(os.environ.get('ERHU_RUNS','work/runs'))
    manifest=pd.read_csv(root/'00_manifest/benchmark_manifest_12.csv')
    frozen=pd.read_csv(root/'00_manifest/frozen_score_events.csv')
    dest=root/'readout';dest.mkdir(parents=True,exist_ok=True)
    for method in ['theglue','dual_dtw']:
        results=[]
        for row in manifest.to_dict('records'):
            cid=row['performance_id']
            a=pd.read_csv(root/'02_alignment'/method/cid/'alignment.csv')
            s=frozen[frozen.performance_id==cid].reset_index(drop=True)
            assert a.score_event_id.tolist()==s.score_note_id.tolist()
            a['score_f0_hz']=s.score_pitch_hz
            matched=a[a.alignment_status=='match'].copy()
            matched['native_index']=matched.performance_note_id.str[1:].astype(int)
            assert matched.native_index.is_unique
            ordered=matched.sort_values(['predicted_anchor_sec','native_index'],kind='stable')
            next_onset=dict(zip(ordered.performance_note_id,ordered.predicted_anchor_sec.shift(-1)))
            p=pd.read_csv(runs/cid/f'{cid}_CREPE_plain_raw.csv')
            t=p['时间(s)'].to_numpy(float);f=p['onset频率(Hz)'].to_numpy(float)
            valid=np.isfinite(f)&(f>0)
            if 'hard_invalid' in p:
                valid &= ~p.hard_invalid.astype(str).str.lower().isin(['true','1','yes']).to_numpy()
            result=evaluate(cid,method,150,(a,t,f,valid,next_onset,{}))
            result.to_csv(dest/f'{method}_{cid}.csv',index=False);results.append(result)
        pd.DataFrame([summary(pd.concat(results),'all')]).to_csv(dest/f'{method}_summary.csv',index=False)

if __name__=='__main__':main()
