from pathlib import Path
import csv as csvlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
import mel_support as e
import dp3_legacy as dp
BEST_ADMISSIBLE = True
T = e.T

def candidates(case,base,pitch,on,curve,score,i):
    left,right=dp._nearest_recovery_anchors(base,i)
    assert left is not None and right is not None
    lo,hi=float(base.loc[left,T]),float(base.loc[right,T])
    predicted,window,source=dp._predict_recovery_time(base,i,left,right)
    broad=max(window,(hi-lo)*1000)
    raw=e.nums(curve,'novelty_raw');z=e.nums(curve,'novelty_norm')
    # All positive local maxima of raw OR normalized novelty. Flat peak
    # plateaus use scipy's single central sample; union deduplicates frame only.
    ir,_=find_peaks(raw);ir=ir[raw[ir]>0]
    iz,_=find_peaks(z);iz=iz[z[iz]>0]
    frames=np.union1d(ir,iz);st=e.scaled(e.nums(curve,'raw_time_s')[frames],on)
    frames=frames[(st>lo)&(st<hi)]
    pr=dict(zip(ir,peak_prominences(raw,ir)[0]));pz=dict(zip(iz,peak_prominences(z,iz)[0]))
    times=e.scaled(e.nums(curve,'raw_time_s')[frames],on)
    pt=e.nums(pitch,'时间(s)');pf=e.nums(pitch,'pitch(Hz)')
    owned=set(base.loc[e.valid(base),'onset_id'].astype(str))
    used_frames=set(pd.to_numeric(on.loc[on.onset_id.astype(str).isin(owned),'note_frame'],errors='coerce').dropna().astype(int))
    sf=float(base.loc[i,'谱面频率(Hz)']);sm=dp.midi_round(e.nums(score,dp.SCORE_F0_COL));score_t=dp.get_score_time_array(score)
    rows=[];pool=[]
    start_index=int(e.nums(on,'original_onset_index').max())+1
    for k,f in enumerate(frames):
        rt=float(curve.raw_time_s.iloc[f]);t=float(times[k]);f0=float(pf[np.argmin(np.abs(pt-rt))])
        cents=dp.cents_error(sf,f0) if np.isfinite(f0) and f0>0 else np.nan
        pitchok=bool(np.isfinite(cents) and abs(cents)<=dp.RECOVERY_NORMAL_PITCH_MAX_CENTS and abs(cents)<=dp.RECOVERY_ABSOLUTE_PITCH_MAX_CENTS)
        order=lo<t<hi;ownership=int(round(rt/.005)) not in used_frames
        dt=(t-predicted)*1000;localtime=abs(dt)<=window;boundtime=abs(dt)<=broad
        source_penalty=dp.RECOVERY_SOURCE_PENALTY.get('mel_spectral',.12)
        lc=.55*abs(dt)/window+.35*abs(cents)/100+.10*source_penalty if np.isfinite(cents) else np.inf
        bc=.55*abs(dt)/broad+.35*abs(cents)/100+.10*source_penalty if np.isfinite(cents) else np.inf
        ot=e.nums(base,T).copy();ot[i]=t
        spacing=all(dp.main_same_note_gap_ok(l,r,l,r,score,score_t,ot,sm) for l,r in [(left,i),(i,right)])
        ident=f'mel_recall_{int(f)}'
        row=dict(mel_candidate_id=ident,raw_frame=int(f),raw_time_s=rt,alignment_time_s=t,
            novelty_strength=float(z[f]),raw_novelty_strength=float(raw[f]),
            prominence=float(pz.get(f,0.)),raw_prominence=float(pr.get(f,0.)),
            peak_origin='raw+normalized' if f in pr and f in pz else ('raw' if f in pr else 'normalized'),
            f0_hz=f0,score_pitch_hz=sf,cents_error=cents,pitch_gate_cents=dp.RECOVERY_NORMAL_PITCH_MAX_CENTS,
            pitch_pass=pitchok,local_timing_pass=bool(localtime),anchor_bounded_timing_pass=bool(boundtime),
            order_pass=bool(order),ownership_pass=bool(ownership),spacing_pass=bool(spacing),
            local_cost=lc,anchor_bounded_cost=bc,local_cost_pass=bool(lc<=dp.RECOVERY_MAX_COST),
            anchor_bounded_cost_pass=bool(bc<=dp.RECOVERY_MAX_COST))
        rows.append(row)
        candidate={col:np.nan for col in on.columns}
        candidate.update({'note_time(s)':t,'note_f0(Hz)':f0,'note_frame':int(round(rt/.005)),
            'pre_time(s)':t,'pre_frame':int(round(rt/.005)),'pre_src':'mel_spectral',
            'note_color':'purple','pre_color':'purple','candidate_level':'strong',
            'candidate_confidence':np.nan,'onset_id':ident,'original_onset_index':start_index+k,
            'study_provenance':'all_positive_raw_or_normalized_Mel_local_maxima',
            'mel_strength':float(z[f]),'mel_raw_strength':float(raw[f]),
            'tempo_scale_anchor_s':on.tempo_scale_anchor_s.iloc[0],'global_scale':on.global_scale.iloc[0]})
        pool.append(candidate)
    diag=pd.DataFrame(rows);pool=pd.DataFrame(pool)
    diag['local_rank']=diag.novelty_strength.rank(ascending=False,method='min').astype(int)
    essential=diag.pitch_pass&diag.order_pass&diag.ownership_pass&diag.spacing_pass
    localgood=essential&diag.local_timing_pass&diag.local_cost_pass
    branch='local_window_primary' if localgood.any() else 'anchor_bounded_fallback'
    diag['timing_pass']=diag.local_timing_pass if localgood.any() else diag.anchor_bounded_timing_pass
    diag['effective_cost']=diag.local_cost if localgood.any() else diag.anchor_bounded_cost
    diag['cost_pass']=diag.local_cost_pass if localgood.any() else diag.anchor_bounded_cost_pass
    diag['final_admissible']=essential&diag.timing_pass&diag.cost_pass
    margin=dp.RECOVERY_MIN_ADVANTAGE if localgood.any() else dp.RECOVERY_BILATERAL_MIN_ADVANTAGE
    qualified=diag[diag.final_admissible].sort_values(['effective_cost','alignment_time_s','mel_candidate_id'])
    winner_gap=float(qualified.effective_cost.iloc[1]-qualified.effective_cost.iloc[0]) if len(qualified)>1 else np.inf
    frozen_margin=margin
    if BEST_ADMISSIBLE: margin=0.0
    diag['frozen_required_margin']=frozen_margin
    diag['selection_policy']='best_admissible' if BEST_ADMISSIBLE else 'frozen_uniqueness'
    diag['runner_up_margin']=winner_gap;diag['required_margin']=margin
    diag['unique_winner']=bool(len(qualified)>0 and winner_gap>=margin)
    diag['final_selected']=False
    diag['rejection_reason']=diag.apply(lambda r:';'.join([name for name,col in [('pitch','pitch_pass'),('timing','timing_pass'),('order','order_pass'),('ownership','ownership_pass'),('spacing','spacing_pass'),('cost','cost_pass')] if not r[col]]) or ('ambiguity' if winner_gap<margin else 'admissible_not_selected'),axis=1)
    eligible=diag.ownership_pass&diag.spacing_pass
    # Existing ownership/spacing filters run before frozen one-event recovery;
    # every candidate, including rejected candidates, remains in diagnostic CSV.
    # User-authorized local wrapper override only; never edit DP source files.
    saved_margins=(dp.RECOVERY_MIN_ADVANTAGE,dp.RECOVERY_BILATERAL_MIN_ADVANTAGE)
    try:
        if BEST_ADMISSIBLE:
            dp.RECOVERY_MIN_ADVANTAGE=0.0
            dp.RECOVERY_BILATERAL_MIN_ADVANTAGE=0.0
        result=dp.recover_unmatched_from_free_onsets(base,score,pool.loc[eligible].copy())
    finally:
        dp.RECOVERY_MIN_ADVANTAGE,dp.RECOVERY_BILATERAL_MIN_ADVANTAGE=saved_margins
    if dp._valid_matched_row(result.loc[i]):
        selected=str(result.loc[i,'onset_id']);diag.loc[diag.mel_candidate_id==selected,'final_selected']=True
        diag.loc[diag.mel_candidate_id==selected,'rejection_reason']='SELECTED'
        assert selected==qualified.iloc[0].mel_candidate_id and winner_gap>=margin
    else:
        assert not(len(qualified) and winner_gap>=margin),'Diagnostic scoring disagrees with frozen recovery'
    # Frozen recovery cannot touch any valid row, including all derived fields.
    pd.testing.assert_frame_equal(base.loc[e.valid(base)],result.loc[e.valid(base),base.columns],check_dtype=False,check_exact=True)
    region=dict(left_index=int(left),right_index=int(right),left_anchor_s=lo,right_anchor_s=hi,
        predicted_s=predicted,local_window_ms=window,anchor_bounded_window_ms=broad,
        prediction_source=source,selected_domain=branch,required_margin=margin,runner_up_margin=winner_gap,
        peaks=len(diag),pitch_pass=int(diag.pitch_pass.sum()),after_pitch_timing_order_ownership=int((essential&diag.timing_pass).sum()),
        admissible=int(diag.final_admissible.sum()),success=bool(dp._valid_matched_row(result.loc[i])))
    mask=(e.scaled(e.nums(curve,'raw_time_s'),on)>lo)&(e.scaled(e.nums(curve,'raw_time_s'),on)<hi)
    stats=[e.summary(e.nums(curve,c)[mask],metric=c) for c in ['novelty_raw','novelty_norm']]
    return result,diag,pool,region,pd.DataFrame(stats)


def write_exact_patch(source,dest,final,indices):
    with open(source,encoding='utf-8-sig',newline='') as f:old=list(csvlib.reader(f))
    old=[r for r in old if r and any(v.strip() for v in r)]
    assert len(old)==len(final)+1
    header=old[0];new=[r.copy() for r in old]
    for i in indices:
        new[i+1].extend(['']*(len(header)-len(new[i+1])))
        for k,col in enumerate(header):
            val=final.at[i,col]
            new[i+1][k]='' if pd.isna(val) else str(val)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with open(dest,'w',encoding='utf-8-sig',newline='') as f:csvlib.writer(f).writerows(new)
    with open(dest,encoding='utf-8-sig',newline='') as f:actual=list(csvlib.reader(f))
    for i in range(len(final)):
        if i not in indices:assert old[i+1]==actual[i+1],f'Passthrough row {i} changed'
    return [dict(score_index=i,score_note_id=final.iloc[i].score_note_id,field=col,old=(old[i+1][k] if k<len(old[i+1]) else ''),new=actual[i+1][k])
            for i in indices for k,col in enumerate(header) if (old[i+1][k] if k<len(old[i+1]) else '')!=actual[i+1][k]]
