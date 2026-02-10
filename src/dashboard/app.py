import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

from src.risk.risk_engine import compute_supplier_kpis, risk_score_supplier
from src.simulation.shock import apply_supplier_shock

DATA_PATH = "data/processed/orders_scored.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def top_metrics(df):
    n_orders = len(df)
    n_suppliers = df["supplier_id"].nunique()
    anomaly_rate = float(df["anomaly_flag"].mean()) if "anomaly_flag" in df.columns else 0.0
    rule_rate = float(df["anomaly_flag_rule"].mean()) if "anomaly_flag_rule" in df.columns else None
    return n_orders, n_suppliers, anomaly_rate, rule_rate


def main():
    st.set_page_config(page_title="Pharma Risk Anomaly Dashboard", layout="wide")
    st.title("Pharma Supply Chain Risk Dashboard")

    df = load_data()

    tabs = st.tabs(["Overview", "Supplier Ranking", "Anomaly Explorer", "Scenario Simulator", "Export"])

    # Overview
    with tabs[0]:
        n_orders, n_suppliers, anomaly_rate, rule_rate = top_metrics(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Orders", f"{n_orders:,}")
        c2.metric("Suppliers", f"{n_suppliers:,}")
        c3.metric("Model Anomaly Rate", f"{anomaly_rate*100:.2f}%")
        if rule_rate is not None:
            c4.metric("Rule Anomaly Rate", f"{rule_rate*100:.2f}%")
        else:
            c4.metric("Rule Anomaly Rate", "-")

        kpis = compute_supplier_kpis(df)
        ranked = risk_score_supplier(kpis)

        st.subheader("Top Suppliers by Risk")
        st.dataframe(ranked[["supplier_id", "risk_score", "risk_rank"]].head(10), use_container_width=True)

    # Supplier Ranking
    with tabs[1]:
        st.subheader("Supplier KPIs + Risk Score")
        kpis = compute_supplier_kpis(df)
        ranked = risk_score_supplier(kpis)
        st.dataframe(ranked, use_container_width=True)

    # Anomaly Explorer
    with tabs[2]:
        st.subheader("Top Anomalies (Model)")
        cols = ["date","order_id","supplier_id","pharmacy_id","product_id","anomaly_score","anomaly_flag","reason_codes"]
        cols = [c for c in cols if c in df.columns]

        only_flagged = st.checkbox("Show only flagged anomalies", value=True)
        view = df.copy()
        if only_flagged and "anomaly_flag" in view.columns:
            view = view[view["anomaly_flag"] == 1]

        top_n = st.slider("Top N by anomaly_score", 50, 500, 200, step=50)
        if "anomaly_score" in view.columns:
            view = view.sort_values("anomaly_score", ascending=False).head(top_n)

        st.dataframe(view[cols], use_container_width=True)

        if "anomaly_flag_rule" in df.columns:
            st.subheader("Top Anomalies (Rule Baseline)")
            cols2 = ["date","order_id","supplier_id","pharmacy_id","product_id","anomaly_flag_rule","reason_codes_rule"]
            cols2 = [c for c in cols2 if c in df.columns]
            view2 = df[df["anomaly_flag_rule"] == 1].copy()
            st.dataframe(view2[cols2].head(300), use_container_width=True)

    # Scenario Simulator
    with tabs[3]:
        st.subheader("Shock Scenario Simulator")

        suppliers = sorted(df["supplier_id"].astype(str).unique().tolist())
        supplier = st.selectbox("Supplier", suppliers, index=0)

        lead_mult = st.slider("Lead time multiplier", 1.0, 3.0, 1.5, step=0.1)
        back_boost = st.slider("Backorder boost (probability)", 0.0, 0.5, 0.10, step=0.01)

        if st.button("Run Scenario"):
            base_kpis = compute_supplier_kpis(df)
            base_rank = risk_score_supplier(base_kpis)

            shocked = apply_supplier_shock(df, supplier_id=supplier, lead_time_multiplier=lead_mult, backorder_boost=back_boost, seed=42)
            shock_kpis = compute_supplier_kpis(shocked)
            shock_rank = risk_score_supplier(shock_kpis)

            before = base_rank[["supplier_id","risk_score","risk_rank"]].rename(
                columns={"risk_score":"risk_score_before","risk_rank":"risk_rank_before"}
            )
            after = shock_rank[["supplier_id","risk_score","risk_rank"]].rename(
                columns={"risk_score":"risk_score_after","risk_rank":"risk_rank_after"}
            )
            delta = before.merge(after, on="supplier_id", how="inner")
            delta["risk_score_delta"] = delta["risk_score_after"] - delta["risk_score_before"]
            delta["risk_rank_delta"] = delta["risk_rank_after"] - delta["risk_rank_before"]
            delta = delta.sort_values("risk_score_delta", ascending=False)

            st.write("Top changes (by risk_score_delta):")
            st.dataframe(delta.head(10), use_container_width=True)

    # Export
    with tabs[4]:
        st.subheader("Export")
        kpis = compute_supplier_kpis(df)
        ranked = risk_score_supplier(kpis)

        st.download_button(
            "Download supplier_risk_rank.csv",
            ranked.to_csv(index=False).encode("utf-8"),
            file_name="supplier_risk_rank.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download orders_scored.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name="orders_scored.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

