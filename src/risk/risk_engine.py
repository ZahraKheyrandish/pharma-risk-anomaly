import numpy as np
import pandas as pd


def compute_supplier_kpis(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()

    for c in ["date", "ship_date", "delivery_date", "requested_delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    df["lead_time"] = (df["delivery_date"] - df["ship_date"]).dt.days
    df["late_flag"] = (df["delivery_date"] > df["requested_delivery_date"]).astype(int)

    g = df.groupby("supplier_id", as_index=False)
    out = g.agg(
        n_orders=("order_id", "count"),
        anomaly_rate_model=("anomaly_flag", "mean"),
        anomaly_rate_rule=("anomaly_flag_rule", "mean") if "anomaly_flag_rule" in df.columns else ("anomaly_flag", "mean"),
        late_rate=("late_flag", "mean"),
        mean_lead_time=("lead_time", "mean"),
        var_lead_time=("lead_time", "var"),
        backorder_rate=("backorder_flag", "mean"),
        cold_chain_rate=("cold_chain_breach", "mean"),
        damage_rate=("damage_flag", "mean"),
        mean_fill_rate=("fill_rate", "mean"),
    )

    out["var_lead_time"] = out["var_lead_time"].fillna(0.0)
    return out


def risk_score_supplier(kpis: pd.DataFrame, weights=None) -> pd.DataFrame:
    w = weights or {
        "anomaly_rate_model": 0.30,
        "late_rate": 0.25,
        "var_lead_time": 0.15,
        "backorder_rate": 0.15,
        "cold_chain_rate": 0.10,
        "damage_rate": 0.05,
    }

    df = kpis.copy()

    def norm(col):
        x = df[col].astype(float).values
        mn, mx = np.nanmin(x), np.nanmax(x)
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    for col in w.keys():
        df[f"{col}_norm"] = norm(col)

    df["risk_score"] = 0.0
    for col, weight in w.items():
        df["risk_score"] += weight * df[f"{col}_norm"]

    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df["risk_rank"] = np.arange(1, len(df) + 1)
    return df
