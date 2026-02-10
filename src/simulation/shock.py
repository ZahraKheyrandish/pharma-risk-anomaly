import numpy as np
import pandas as pd


def apply_supplier_shock(
    scored: pd.DataFrame,
    supplier_id: str,
    days: int = 10,
    lead_time_multiplier: float = 1.5,
    backorder_boost: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = scored.copy()

    for c in ["ship_date", "delivery_date", "requested_delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    mask = df["supplier_id"].astype(str) == str(supplier_id)

    if "ship_date" in df.columns and "delivery_date" in df.columns:
        lead = (df.loc[mask, "delivery_date"] - df.loc[mask, "ship_date"]).dt.days
        lead = lead.fillna(0).astype(int)
        lead_shocked = np.ceil(lead * lead_time_multiplier).astype(int)

        df.loc[mask, "delivery_date"] = df.loc[mask, "ship_date"] + pd.to_timedelta(lead_shocked, unit="D")

    if "backorder_flag" in df.columns:
        flip = rng.random(mask.sum()) < backorder_boost
        idx = df.index[mask]
        df.loc[idx[flip], "backorder_flag"] = 1

    df["scenario"] = "shock"
    df["shock_supplier_id"] = str(supplier_id)
    df["shock_days"] = int(days)
    return df
