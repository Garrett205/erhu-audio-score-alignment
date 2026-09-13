import numpy as np
import pandas as pd
VALID = {"对齐成功", "recovered_unmatched"}
Q = [10, 25, 50, 75, 90, 95, 99]
T = "红点时间(s)"

def valid(df):
    return df['对齐状态'].astype(str).str.strip().isin(VALID).to_numpy()


def nums(df, col):
    return pd.to_numeric(df[col], errors='coerce').to_numpy(float)


def scaled(t, onset):
    a, s = float(onset.tempo_scale_anchor_s.iloc[0]), float(onset.global_scale.iloc[0])
    return a + (np.asarray(t) - a) * s


def summary(values, **keys):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    return dict(keys, count=len(v), median=float(np.median(v)) if len(v) else np.nan,
                **{f'P{q}': float(np.percentile(v, q)) if len(v) else np.nan for q in Q})
