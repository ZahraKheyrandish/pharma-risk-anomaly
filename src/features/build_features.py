import pandas as pd
import numpy as np


def build_features(orders: pd.DataFrame, shipments: pd.DataFrame) -> pd.DataFrame:
    o = orders.copy()
    s = shipments.copy()

    o["date"] = pd.to_datetime(o["date"])
    o["requested_delivery_date"] = pd.to_datetime(o["requested_delivery_date"])
    s["ship_date"] = pd.to_datetime(s["ship_date"])
    s["delivery_date"] = pd.to_datetime(s["delivery_date"])

    df = o.merge(
        s[["order_id", "ship_date", "delivery_date", "delivered_qty", "route_id", "cold_chain_ok", "damage_flag"]],
        on="order_id",
        how="left",
    )

    df["lead_time"] = (df["delivery_date"] - df["ship_date"]).dt.days
    df["on_time"] = (df["delivery_date"] <= df["requested_delivery_date"]).astype(int)

    df["fill_rate"] = (df["delivered_qty"] / df["order_qty"]).replace([np.inf, -np.inf], np.nan)
    df["fill_rate"] = df["fill_rate"].clip(lower=0, upper=1).fillna(0)

    df["backorder_flag"] = (df["delivered_qty"] < df["order_qty"]).astype(int)
    df["cold_chain_breach"] = (df["cold_chain_ok"] == 0).astype(int)
    df["damage_flag"] = df["damage_flag"].fillna(0).astype(int)

    df = df.sort_values(["pharmacy_id", "product_id", "date"])

    grp = df.groupby(["pharmacy_id", "product_id"], sort=False)

    df["qty_mean_7d"] = grp["order_qty"].transform(lambda x: x.rolling(7, min_periods=3).mean())
    df["qty_mean_30d"] = grp["order_qty"].transform(lambda x: x.rolling(30, min_periods=7).mean())
    df["qty_std_30d"] = grp["order_qty"].transform(lambda x: x.rolling(30, min_periods=7).std())

    df["qty_z_30d"] = (df["order_qty"] - df["qty_mean_30d"]) / (df["qty_std_30d"] + 1e-6)

    df["qty_q995_30d"] = grp["order_qty"].transform(lambda x: x.rolling(30, min_periods=10).quantile(0.995))
    df["qty_spike_flag"] = (df["order_qty"] >= df["qty_q995_30d"]).astype(int)

    grp2 = df.groupby("supplier_id", sort=False)
    df["lt_mean_30d_supplier"] = grp2["lead_time"].transform(lambda x: x.rolling(30, min_periods=10).mean())
    df["lt_std_30d_supplier"] = grp2["lead_time"].transform(lambda x: x.rolling(30, min_periods=10).std())
    df["lt_z_30d_supplier"] = (df["lead_time"] - df["lt_mean_30d_supplier"]) / (df["lt_std_30d_supplier"] + 1e-6)

    df["lt_q995_30d_supplier"] = grp2["lead_time"].transform(lambda x: x.rolling(30, min_periods=15).quantile(0.995))
    df["late_flag_rule"] = (df["lead_time"] >= df["lt_q995_30d_supplier"]).astype(int)

    fill_cols = [
        "qty_mean_7d",
        "qty_mean_30d",
        "qty_std_30d",
        "qty_z_30d",
        "qty_q995_30d",
        "qty_spike_flag",
        "lt_mean_30d_supplier",
        "lt_std_30d_supplier",
        "lt_z_30d_supplier",
        "lt_q995_30d_supplier",
        "late_flag_rule",
        "lead_time",
    ]

    for c in fill_cols:
        if c in ["qty_spike_flag", "late_flag_rule"]:
            df[c] = df[c].fillna(0).astype(int)
        else:
            df[c] = df[c].fillna(df[c].median())

    return df
