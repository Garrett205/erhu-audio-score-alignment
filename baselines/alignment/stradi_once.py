"""Invoke official CLI unchanged; persist the original predict return value."""
from pathlib import Path
import sys, runpy, json, hashlib, contextlib, time, faulthandler
import pandas as pd
import numpy as np
OUT=Path(__import__("os").environ.get("ERHU_BASELINE_WORK", "work/external")).resolve()
REPO=Path(__import__("os").environ["STRADI_REPO"]).resolve()
sys.path.insert(0,str(REPO))
import violin_transcription.inference as inference
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
original_predict=inference.predict
checkpoint=REPO/'checkpoints/violin_transcription_offline.pth'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()=='2B344F91FE5D49499FAF553B982AB5944EE6B9BE46DE07879CA727FA54F64123'
for row in pd.read_csv(OUT/'00_manifest/benchmark_manifest_12.csv').to_dict('records'):
    cid=row['performance_id']
    dest=OUT/'01_stradi'/cid
    dest.mkdir(parents=True,exist_ok=True)
    done=dest/'complete.json'
    if done.exists():
        saved=json.loads(done.read_text())
        for name, digest in saved['hashes'].items():
            assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest
        print('REUSE',cid,flush=True)
        continue
    if (dest/'native_outputs.npz').exists():
        raise RuntimeError(f'{cid}: native inference exists without completion; recover outputs without retranscription')
    def capture(*args,**kwargs):
        print('NATIVE_INFERENCE_BEGIN',flush=True)
        result=original_predict(*args,**kwargs)
        print('NATIVE_INFERENCE_RETURNED',flush=True)
        np.savez_compressed(dest/'native_outputs.npz',**result)
        (dest/'native_output_shapes.json').write_text(json.dumps({k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in result.items()},indent=2))
        return result
    inference.predict=capture
    sys.argv=[str(REPO/'transcribe.py'),row['wav_path'],'--checkpoint',str(checkpoint),'--model','offline','--output-dir',str(dest)]
    print('START',cid,flush=True)
    start=time.time()
    with (dest/'run.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        faulthandler.dump_traceback_later(60,repeat=True,file=log)
        runpy.run_path(str(REPO/'transcribe.py'),run_name='__main__')
        faulthandler.cancel_dump_traceback_later()
    for ext in ('csv','mid'):
        (dest/f'{cid}.{ext}').rename(dest/f'transcription.{ext}')
    meta=dict(performance_id=cid,notes=len(pd.read_csv(dest/'transcription.csv')),elapsed_seconds=time.time()-start,thresholds=dict(onset=0.5,frame=0.5,offset=0.5),hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.suffix in ('.npz','.csv','.mid')})
    done.write_text(json.dumps(meta,indent=2))
    print('DONE',cid,meta['notes'],round(meta['elapsed_seconds'],2),flush=True)
