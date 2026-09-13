"""Synthetic invariants only; no dataset files or neural inference."""
from pathlib import Path
import os,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src/pipeline'),str(ROOT/'baselines/alignment')]
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import pandas as pd
import pitch,onset,mxml,dp1_legacy,dp2_legacy,dp3_legacy
import mel_rescue,mel_onset,anchor_readout

def main():
    # Gate strict threshold edges and the end-gate negative back offset.
    f=np.zeros(300);q=np.zeros(300);f[100:200]=440;q[100:200]=.5
    assert onset.find_start_allow_frame(f,q,len(f))==60
    assert onset.find_end_cut_frame(f,q,len(f))==198
    assert onset.find_start_allow_frame(np.full(100,1800.),np.ones(100),100)==0
    assert onset.find_end_cut_frame(np.full(100,440.),np.full(100,.01),100)==99
    # Short closed octave islands are repaired, invalid frames remain invalid.
    x=np.r_[np.full(20,440.),np.full(5,880.),np.full(20,440.)]
    repaired=pitch.repair_short_octave_islands(x)
    if isinstance(repaired,tuple):repaired=repaired[0]
    assert np.allclose(repaired,440)
    f=np.full(100,440.);peri=np.ones(100);db=np.full(100,30.)
    peri[30:40]=0;db[60:70]=0
    fused=pitch.fuse_pitch_tracks(f,peri,f,db)
    assert (fused['onset频率(Hz)'].iloc[30:40]==0).all()
    assert (fused['onset频率(Hz)'].iloc[60:70]==0).all()
    # Score IDs are stable across measures, chords and a tie chain.
    m=mxml.m21;s=m.stream.Score();part=m.stream.Part()
    measure=m.stream.Measure(number=1)
    a=m.note.Note('A4',quarterLength=1);a.tie=m.tie.Tie('start');measure.append(a)
    b=m.note.Note('A4',quarterLength=1);b.tie=m.tie.Tie('stop');measure.append(b)
    measure.append(m.chord.Chord(['C5','E5'],quarterLength=1));part.append(measure)
    measure2=m.stream.Measure(number=2)
    measure2.append(m.note.Note('D5').getGrace());measure2.append(m.note.Note('E5'))
    part.append(measure2);s.append(part)
    table,_,_=mxml.make_note_rows(s,mxml.build_tempo_map(s),{})
    assert table.score_note_id.is_unique and len(table)==6
    assert table.score_note_id.iloc[2].endswith('_C01')
    assert table.score_note_id.iloc[3].endswith('_C02')
    legacy=mxml.make_legacy_table(table,False)
    assert '参与起音对齐' in legacy and not any('技巧' in c for c in legacy)
    assert legacy['参与起音对齐'].iloc[1]=='否'
    # Simple ordered DP correspondence and readout boundary behavior.
    f0=np.array([440.,493.883,523.251]); midi=np.array([69,71,72])
    result=dp1_legacy.build_dp_alignment(midi,midi,f0,f0,np.arange(3.),np.arange(3.))
    assert [(r[0],r[1]) for r in result]==[(0,0),(1,1),(2,2)]
    anchor_readout.tests()
    assert mel_rescue.BEST_ADMISSIBLE is True
    print('PASS: gate edges, pitch invalidity/octave repair, score IDs/ties/chords/grace, DP1, readout, Mel policy')

if __name__=='__main__':main()
