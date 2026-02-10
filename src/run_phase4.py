import os
import pandas as pd

from src.risk.risk_engine import compute_supplier_kpis, risk_score_supplier
from src.simulation.shock import apply_supplier_shock

IN_PATH = "data/processed/orders_scored.csv"
OUT_DIR = "data/processed"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base = pd.read_csv(IN_PATH)
    base["scenario"] = "base"

    kpis_base = compute_supplier_kpis(base)
    rank_base = risk_score_supplier(kpis_base)

    kpis_base.to_csv(f"{OUT_DIR}/supplier_kpis.csv", index=False)
    rank_base.to_csv(f"{OUT_DIR}/supplier_risk_rank.csv", index=False)

    top_supplier = str(rank_base.iloc[0]["supplier_id"])
    shocked = apply_supplier_shock(base, supplier_id=top_supplier, days=10, lead_time_multiplier=1.5, backorder_boost=0.10, seed=42)

    kpis_shock = compute_supplier_kpis(shocked)
    rank_shock = risk_score_supplier(kpis_shock)

    before = rank_base[["supplier_id", "risk_score", "risk_rank"]].copy()
    after = rank_shock[["supplier_id", "risk_score", "risk_rank"]].copy()
    before = before.rename(columns={"risk_score": "risk_score_before", "risk_rank": "risk_rank_before"})
    after = after.rename(columns={"risk_score": "risk_score_after", "risk_rank": "risk_rank_after"})

    delta = before.merge(after, on="supplier_id", how="inner")
    delta["risk_score_delta"] = delta["risk_score_after"] - delta["risk_score_before"]
    delta["risk_rank_delta"] = delta["risk_rank_after"] - delta["risk_rank_before"]

    delta = delta.sort_values("risk_score_delta", ascending=False).reset_index(drop=True)
    delta.to_csv(f"{OUT_DIR}/scenario_before_after.csv", index=False)

    print("[DONE] Saved:")
    print(" - data/processed/supplier_kpis.csv")
    print(" - data/processed/supplier_risk_rank.csv")
    print(" - data/processed/scenario_before_after.csv")
    print("Shock supplier:", top_supplier)


if __name__ == "__main__":
    main()
