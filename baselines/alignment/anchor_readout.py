import numpy as np
import pandas as pd
import pitch_onset_dtw as core

def first_anchor(t,f,valid,target,start,end):
    """Earliest pair of adjacent original 5-ms frames, both within closed interval."""
    indices=np.flatnonzero((t>=start-1e-9)&(t<=end+1e-9))
    if len(indices)<2:return None
    cents=core.cents_error(f[indices],np.full(len(indices),target))
    good=valid[indices]&np.isfinite(cents)&(np.abs(cents)<=100)
    pair=good[:-1]&good[1:]&(np.diff(indices)==1)&np.isclose(np.diff(t[indices]),.005,rtol=0,atol=1e-9)
    hits=np.flatnonzero(pair)
    if not len(hits):return None
    j=int(hits[0]);return int(indices[j]),int(indices[j+1]),float(cents[j]),float(cents[j+1])


def tests():
    t=np.arange(5)*.005;f=np.array([440.,440.,0.,440.,440.]);v=f>0
    assert first_anchor(t,f,v,440,0,.005)[:2]==(0,1)
    assert first_anchor(t,f,v,440,0,.004) is None
    assert first_anchor(t,f,v,440,.006,.020)[:2]==(3,4)
    assert first_anchor(np.array([0.,.010]),np.array([440.,440.]),np.array([True,True]),440,0,.02) is None
    assert first_anchor(t,f,np.zeros(5,dtype=bool),440,0,.02) is None
    assert first_anchor(t,f,v,880,0,.02) is None


def evaluate(cid,method,w,data):
    a,t,f,valid,next_onset,hashes=data;rows=[]
    for e in a.to_dict('records'):
        r=dict(method=method,performance_id=cid,score_event_id=e['score_event_id'],score_index=e['score_index'],score_pitch=e['score_pitch'],score_f0_hz=e['score_f0_hz'],performance_note_id=e['performance_note_id'],native_alignment_status=e['alignment_status'],stradi_onset=e['predicted_anchor_sec'],stradi_offset=e['performance_offset_sec'],W_ms=w,status='FAIL',reason='native_unmatched',predicted_anchor_sec=None,next_matched_performance_onset=None,search_end=None,anchor_frame_index=None,second_frame_index=None,anchor_f0_hz=None,anchor_cents=None,second_frame_time=None,second_f0_hz=None,second_cents=None)
        if e['alignment_status']=='match':
            start=float(e['predicted_anchor_sec']);end=float(e['performance_offset_sec']);nxt=next_onset[e['performance_note_id']]
            bound=min(end,start+w/1000,nxt if np.isfinite(nxt) else float('inf'))
            r.update(next_matched_performance_onset=nxt if np.isfinite(nxt) else None,search_end=bound,reason='no_two_consecutive_valid_100c_frames')
            hit=first_anchor(t,f,valid,e['score_f0_hz'],start,bound)
            if hit:
                i,j,c1,c2=hit;r.update(status='PASS',reason='earliest_two_frame_100c_anchor',predicted_anchor_sec=float(t[i]),anchor_frame_index=i,second_frame_index=j,anchor_f0_hz=float(f[i]),anchor_cents=c1,second_frame_time=float(t[j]),second_f0_hz=float(f[j]),second_cents=c2)
                assert start-1e-9<=t[i]<t[j]<=bound+1e-9
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df,split):
    matched=df.native_alignment_status.eq('match');passed=df.status.eq('PASS');n=len(df)
    return dict(method=df.method.iloc[0],split=split,total=n,passed=int(passed.sum()),pass_percent=100*passed.sum()/n,matched=int(matched.sum()),unmatched=int((~matched).sum()),matched_with_valid_anchor=int((matched&passed).sum()),matched_but_no_100c_anchor=int((matched&~passed).sum()))
