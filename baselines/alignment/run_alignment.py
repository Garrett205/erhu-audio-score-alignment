"""No proposed acoustic data or evaluator imported in this process."""
from pathlib import Path
import sys, json, hashlib, time, contextlib, traceback, argparse
import numpy as np
import pandas as pd
OUT=Path(__import__("os").environ.get("ERHU_BASELINE_WORK", "work/external")).resolve()
parser=argparse.ArgumentParser(); parser.add_argument('method',choices=['dual_dtw','audio_to_score','theglue']); args=parser.parse_args()
from parangonar.match import DualDTWNoteMatcher, AudioToScoreMatcher, TheGlueNoteMatcher
matcher={'dual_dtw':DualDTWNoteMatcher,'audio_to_score':AudioToScoreMatcher,'theglue':TheGlueNoteMatcher}[args.method]()
for row in pd.read_csv(OUT/'00_manifest/benchmark_manifest_12.csv').to_dict('records'):
    cid=row['performance_id']; dest=OUT/'02_alignment'/args.method/cid; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        print('REUSE',args.method,cid,flush=True); continue
    sna=np.load(OUT/'00_manifest/symbolic'/f'{cid}.npy')
    print('START',args.method,cid,flush=True)
    start=time.time()
    try:
        with (dest/'run.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if args.method=='audio_to_score':
                import soundfile as sf
                audio,sr=sf.read(row['wav_path'])
                alignment,D,onsets,spec,path=matcher(audio,sna,sample_rate=sr,return_everything=True)
                np.savez_compressed(dest/'native_acoustic_evidence.npz',onsets=onsets,spectrogram=spec,path=path)
                pna=None
            else:
                source=OUT/'01_stradi'/cid
                meta=json.loads((source/'complete.json').read_text())
                assert hashlib.sha256((source/'transcription.csv').read_bytes()).hexdigest()==meta['hashes']['transcription.csv']
                p=pd.read_csv(source/'transcription.csv')
                pna=np.zeros(len(p),dtype=[('id','U128'),('pitch','i4'),('onset_sec','f8'),('duration_sec','f8'),('velocity','i4')])
                pna['id']=[f'p{i:06d}' for i in range(len(p))]; pna['pitch']=p.midi_number
                pna['onset_sec']=p.onset; pna['duration_sec']=p.offset-p.onset; pna['velocity']=80
                alignment=matcher(sna,pna)
            (dest/'alignment_native.json').write_text(json.dumps(alignment,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
            maps={}; insertions=[]
            for event in alignment:
                if event['label']=='insertion': insertions.append(event); continue
                sid=str(event['score_id']); assert sid not in maps, sid; maps[sid]=event
            assert not (set(maps)-set(sna['id']))
            output=[]; pmap={} if pna is None else {str(n['id']):n for n in pna}
            for idx,s in enumerate(sna):
                e=maps.get(str(s['id']),{'label':'deletion'})
                matched=e['label']=='match'
                pn=pmap.get(str(e.get('performance_id','')))
                anchor=float(e['performance_time']) if matched and pna is None else (float(pn['onset_sec']) if matched else None)
                output.append(dict(performance_id=cid,score_event_id=str(s['id']),score_index=idx,score_pitch=int(s['pitch']),score_pitch_midi=int(s['pitch']),score_symbolic_onset=float(s['onset_beat']),predicted_anchor_sec=anchor,matched_performance_onset=anchor,alignment_status='match' if matched else 'unmatched',performance_note_id=e.get('performance_id'),performance_pitch_midi=int(pn['pitch']) if pn is not None else None,performance_offset_sec=float(pn['onset_sec']+pn['duration_sec']) if pn is not None else None))
            pd.DataFrame(output).to_csv(dest/'alignment.csv',index=False)
            (dest/'alignment.json').write_text(json.dumps(output,indent=2))
            (dest/'insertions.json').write_text(json.dumps(insertions,indent=2))
            nmatch=sum(e['alignment_status']=='match' for e in output)
            stats=dict(method=args.method,performance_id=cid,N_score=len(sna),N_matched=nmatch,N_unmatched=len(sna)-nmatch,N_insertions=len(insertions) if pna is not None else None,N_performance_notes=len(pna) if pna is not None else None,elapsed_seconds=time.time()-start)
            (dest/'complete.json').write_text(json.dumps(stats,indent=2))
        print('DONE',stats,flush=True)
    except Exception:
        (dest/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('FAILED',cid,traceback.format_exc(),flush=True)
