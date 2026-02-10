import os
import pandas as pd

from src.scoring.validate import validate_orders
from src.features.build_features import build_features
from src.models.score_orders import score_orders_train_test

DATA_DIR = "data/synth"
OUT_DIR = "data/processed"


def _rule_reason_codes(row):
    reasons = []
    if row.get("cold_chain_breach", 0) == 1:
        reasons.append("COLD_CHAIN")
    if row.get("damage_flag", 0) == 1:
        reasons.append("DAMAGE")
    if row.get("backorder_flag", 0) == 1:
        reasons.append("BACKORDER")
    if row.get("qty_spike_flag", 0) == 1:
        reasons.append("QTY_SPIKE")
    if row.get("late_flag_rule", 0) == 1:
        reasons.append("LATE_DELIVERY")
    return "|".join(reasons)


def _print_metrics(df, pred_col, true_col, prefix):
    tp = ((df[pred_col] == 1) & (df[true_col] == 1)).sum()
    pred_pos = (df[pred_col] == 1).sum()
    true_pos = (df[true_col] == 1).sum()

    precision = tp / max(1, pred_pos)
    recall = tp / max(1, true_pos)
    overlap = tp / max(1, len(df))

    print(f"{prefix} precision:", float(precision))
    print(f"{prefix} recall:", float(recall))
    print(f"{prefix} overlap:", float(overlap))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    orders = pd.read_csv(f"{DATA_DIR}/fact_orders_daily.csv")
    shipments = pd.read_csv(f"{DATA_DIR}/fact_shipments.csv")

    validated = validate_orders(orders)
    validated.to_csv(f"{OUT_DIR}/orders_validated.csv", index=False)

    valid_orders = validated[validated["is_valid"]].copy()

    feats = build_features(valid_orders, shipments)
    feats.to_csv(f"{OUT_DIR}/orders_features.csv", index=False)

    scored = score_orders_train_test(feats, contamination="auto", seed=42, test_size=0.3)

    scored["anomaly_flag_rule"] = (
        (scored["qty_spike_flag"] == 1)
        | (scored["late_flag_rule"] == 1)
        | (scored["cold_chain_breach"] == 1)
        | (scored["backorder_flag"] == 1)
    ).astype(int)

    scored["reason_codes_rule"] = scored.apply(_rule_reason_codes, axis=1)

    scored.to_csv(f"{OUT_DIR}/orders_scored.csv", index=False)

    print("[DONE] Saved to data/processed/")
    print("Contamination used:", float(scored["contamination_used"].iloc[0]))
    print("Train threshold:", float(scored["threshold_train"].iloc[0]))

    test = scored[scored["split"] == "test"]
    print("Test anomaly rate (model):", float(test["anomaly_flag"].mean()))
    print("Test anomaly rate (rule):", float(test["anomaly_flag_rule"].mean()))

    if "anomaly_flag_gen" in test.columns:
        print("Test gen rate:", float(test["anomaly_flag_gen"].mean()))
        _print_metrics(test, "anomaly_flag", "anomaly_flag_gen", "Model")
        _print_metrics(test, "anomaly_flag_rule", "anomaly_flag_gen", "Rule")


if __name__ == "__main__":
    main()
