import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    "order_qty",
    "unit_price",
    "lead_time",
    "fill_rate",
    "backorder_flag",
    "cold_chain_breach",
    "damage_flag",
    "qty_z_30d",
    "lt_z_30d_supplier",
    "qty_spike_flag",
    "late_flag_rule",
]


def make_reason_codes(row):
    reasons = []
    if row.get("cold_chain_breach", 0) == 1:
        reasons.append("COLD_CHAIN")
    if row.get("damage_flag", 0) == 1:
        reasons.append("DAMAGE")
    if row.get("backorder_flag", 0) == 1:
        reasons.append("BACKORDER")
    if row.get("qty_spike_flag", 0) == 1 or abs(row.get("qty_z_30d", 0.0)) >= 3.0:
        reasons.append("QTY_SPIKE")
    if row.get("late_flag_rule", 0) == 1 or row.get("lt_z_30d_supplier", 0.0) >= 3.0:
        reasons.append("LATE_DELIVERY")
    return "|".join(reasons)


def _resolve_contamination(df, contamination):
    if contamination == "auto":
        if "anomaly_flag_gen" in df.columns:
            contamination = float(df["anomaly_flag_gen"].mean())
        else:
            contamination = 0.01
    return float(np.clip(float(contamination), 0.001, 0.5))


def score_orders_train_test(df: pd.DataFrame, contamination="auto", seed=42, test_size=0.3) -> pd.DataFrame:
    out = df.copy()

    idx = np.arange(len(out))
    strat = out["anomaly_flag_gen"] if "anomaly_flag_gen" in out.columns else None
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)

    out["split"] = "train"
    out.loc[te_idx, "split"] = "test"

    contam = _resolve_contamination(out.iloc[tr_idx], contamination)

    X_tr = out.iloc[tr_idx][FEATURE_COLS].astype(float).values
    X_te = out.iloc[te_idx][FEATURE_COLS].astype(float).values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = IsolationForest(n_estimators=400, contamination=contam, random_state=seed, n_jobs=-1)
    model.fit(X_tr_s)

    score_tr = -model.score_samples(X_tr_s)
    score_te = -model.score_samples(X_te_s)

    thr = float(np.quantile(score_tr, 1.0 - contam))

    out["anomaly_score"] = np.nan
    out["anomaly_flag"] = 0

    out.iloc[tr_idx, out.columns.get_loc("anomaly_score")] = score_tr
    out.iloc[te_idx, out.columns.get_loc("anomaly_score")] = score_te

    out.iloc[tr_idx, out.columns.get_loc("anomaly_flag")] = (score_tr >= thr).astype(int)
    out.iloc[te_idx, out.columns.get_loc("anomaly_flag")] = (score_te >= thr).astype(int)

    out["reason_codes"] = out.apply(make_reason_codes, axis=1)
    out["contamination_used"] = contam
    out["threshold_train"] = thr
    return out
