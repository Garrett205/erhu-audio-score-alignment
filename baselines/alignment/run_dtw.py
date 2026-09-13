"""I/O adapter; preserves frozen baseline configurations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pitch_dtw
import pitch_onset_dtw

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['pitch','pitch-onset'],required=True)
    p.add_argument('--score',type=Path,required=True)
    p.add_argument('--pitch',type=Path,required=True)
    p.add_argument('--onset',type=Path)
    p.add_argument('--input',type=Path,help='WAV if no onset-strength CSV is supplied')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.method=='pitch-onset' and a.onset is None and a.input is None:
        p.error('pitch-onset requires --onset or --input')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.method=='pitch':
        result= pitch_dtw.run_piece(a.score,a.pitch,'input')
        summary=result[-1].to_dict()
    else:
        result= pitch_onset_dtw.run_piece(a.score,a.pitch,'input',a.onset,a.input)
        summary=result[-2].to_dict()
    result[0].to_csv(a.output/'predictions.csv',index=False)
    np.save(a.output/'path.npy',result[1])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
