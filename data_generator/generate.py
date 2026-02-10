import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OUT_DIR = "data/synth"

SEED = 42
N_DAYS = 180
N_SUPPLIERS = 8
N_PHARMACIES = 50
N_PRODUCTS = 30

ANOMALY_RATE_TARGET = 0.01
COLD_CHAIN_RATIO = 0.25


def daterange(start_date, n_days):
    return [start_date + timedelta(days=i) for i in range(n_days)]


def make_dims(rng):
    suppliers = pd.DataFrame({
        "supplier_id": [f"S{i:02d}" for i in range(1, N_SUPPLIERS + 1)],
        "supplier_name": [f"Supplier_{i:02d}" for i in range(1, N_SUPPLIERS + 1)],
        "country": rng.choice(["IR", "TR", "AE", "CN", "IN"], size=N_SUPPLIERS),
        "base_capacity_units_per_day": rng.integers(800, 2500, size=N_SUPPLIERS),
        "quality_score_base": np.round(rng.uniform(0.85, 0.99, size=N_SUPPLIERS), 3),
    })

    tiers = rng.choice(["A", "B", "C"], p=[0.2, 0.5, 0.3], size=N_PHARMACIES)
    mult = np.where(
        tiers == "A", rng.uniform(1.4, 2.2, N_PHARMACIES),
        np.where(tiers == "B", rng.uniform(0.9, 1.4, N_PHARMACIES),
                 rng.uniform(0.5, 0.9, N_PHARMACIES))
    )

    pharmacies = pd.DataFrame({
        "pharmacy_id": [f"P{i:03d}" for i in range(1, N_PHARMACIES + 1)],
        "city": rng.choice(["Tehran", "Mashhad", "Shiraz", "Isfahan", "Tabriz"], size=N_PHARMACIES),
        "tier": tiers,
        "demand_multiplier": np.round(mult, 3),
    })

    is_cold = (rng.random(N_PRODUCTS) < COLD_CHAIN_RATIO).astype(int)
    products = pd.DataFrame({
        "product_id": [f"DRUG{i:03d}" for i in range(1, N_PRODUCTS + 1)],
        "product_name": [f"Drug_{i:03d}" for i in range(1, N_PRODUCTS + 1)],
        "is_cold_chain": is_cold,
        "shelf_life_days": rng.integers(180, 900, size=N_PRODUCTS),
    })

    return suppliers, pharmacies, products


def seasonality_factor(day_of_year):
    yearly = 1.0 + 0.20 * np.sin(2 * np.pi * day_of_year / 365.0)
    weekly = 1.0 + 0.08 * np.sin(2 * np.pi * (day_of_year % 7) / 7.0)
    return yearly * weekly


def generate_orders_shipments_inventory(suppliers, pharmacies, products, rng):
    start_date = datetime.today().date() - timedelta(days=N_DAYS)
    dates = daterange(start_date, N_DAYS)

    prod_to_supplier = {pid: rng.choice(suppliers["supplier_id"].values) for pid in products["product_id"].values}

    orders, shipments, inv_rows = [], [], []
    inv = {(p, d): int(rng.integers(50, 200)) for p in pharmacies["pharmacy_id"] for d in products["product_id"]}

    order_counter = 1
    shipment_counter = 1

    shock_supplier = rng.choice(suppliers["supplier_id"].values)
    shock_start_idx = int(N_DAYS * 0.55)
    shock_len = 10
    shock_days = set(dates[shock_start_idx: shock_start_idx + shock_len])

    for dt in dates:
        doy = dt.timetuple().tm_yday
        sf = seasonality_factor(doy)

        for _, ph in pharmacies.iterrows():
            base_lambda = 1.2 * float(ph["demand_multiplier"]) * sf
            n_orders = rng.poisson(lam=max(0.2, base_lambda))

            for _ in range(n_orders):
                prod = products.sample(1, random_state=int(rng.integers(0, 1e9))).iloc[0]
                product_id = prod["product_id"]
                supplier_id = prod_to_supplier[product_id]

                qty = int(max(1, np.round(rng.lognormal(mean=3.0, sigma=0.5))))
                unit_price = float(np.round(rng.uniform(1.0, 20.0), 2))

                req_days = int(rng.integers(2, 7))
                requested_delivery_date = (dt + timedelta(days=req_days))

                order_id = f"O{order_counter:07d}"
                order_counter += 1

                order_channel = rng.choice(["web", "phone", "api"], p=[0.6, 0.25, 0.15])
                is_priority = int(rng.random() < 0.08)

                orders.append({
                    "date": dt.isoformat(),
                    "order_id": order_id,
                    "pharmacy_id": ph["pharmacy_id"],
                    "supplier_id": supplier_id,
                    "product_id": product_id,
                    "order_qty": qty,
                    "unit_price": unit_price,
                    "requested_delivery_date": requested_delivery_date.isoformat(),
                    "order_channel": order_channel,
                    "is_priority": is_priority,
                })

                ship_date = dt + timedelta(days=int(rng.random() < 0.25))

                base_lt = int(rng.integers(1, 5))
                in_shock = (supplier_id == shock_supplier) and (dt in shock_days)
                lt = base_lt + (int(rng.integers(3, 7)) if in_shock else int(rng.integers(0, 3)))
                delivery_date = ship_date + timedelta(days=lt)

                delivered_qty = qty
                if in_shock and rng.random() < 0.35:
                    delivered_qty = int(max(0, np.round(qty * rng.uniform(0.3, 0.8))))
                elif rng.random() < 0.05:
                    delivered_qty = int(max(0, np.round(qty * rng.uniform(0.6, 0.95))))

                cold_chain_ok = 1
                if int(prod["is_cold_chain"]) == 1 and rng.random() < 0.01:
                    cold_chain_ok = 0

                damage_flag = int(rng.random() < 0.006)
                route_id = f"R{int(rng.integers(1, 15)):02d}"

                shipment_id = f"SH{shipment_counter:07d}"
                shipment_counter += 1

                shipments.append({
                    "shipment_id": shipment_id,
                    "order_id": order_id,
                    "ship_date": ship_date.isoformat(),
                    "delivery_date": delivery_date.isoformat(),
                    "delivered_qty": delivered_qty,
                    "route_id": route_id,
                    "cold_chain_ok": cold_chain_ok,
                    "damage_flag": damage_flag,
                })

        # inventory daily (ساده)
        for ph_id in pharmacies["pharmacy_id"].values:
            tier = pharmacies.loc[pharmacies["pharmacy_id"] == ph_id, "tier"].values[0]
            sales_mu = 3 if tier == "A" else (2 if tier == "B" else 1)

            for prod_id in products["product_id"].values:
                key = (ph_id, prod_id)
                starting = inv[key]
                daily_sales = int(rng.poisson(lam=sales_mu))
                arrivals = int(rng.poisson(lam=0.5))
                ending = max(0, starting + arrivals - daily_sales)
                stockout_flag = int(ending == 0)

                inv_rows.append({
                    "date": dt.isoformat(),
                    "pharmacy_id": ph_id,
                    "product_id": prod_id,
                    "starting_inventory": starting,
                    "ending_inventory": ending,
                    "daily_sales": daily_sales,
                    "stockout_flag": stockout_flag
                })
                inv[key] = ending

    orders_df = pd.DataFrame(orders)
    shipments_df = pd.DataFrame(shipments)
    inv_df = pd.DataFrame(inv_rows)

    # anomaly ground truth (generator-level)
    lead_time = (pd.to_datetime(shipments_df["delivery_date"]) - pd.to_datetime(shipments_df["ship_date"])).dt.days
    tmp = orders_df[["order_id", "order_qty"]].merge(
        shipments_df[["order_id", "delivered_qty", "cold_chain_ok"]].assign(lead_time=lead_time),
        on="order_id",
        how="left"
    )

    qty_spike = tmp["order_qty"] > tmp["order_qty"].quantile(0.995)
    late = tmp["lead_time"] > tmp["lead_time"].quantile(0.995)
    cold = (tmp["cold_chain_ok"] == 0)
    backorder = (tmp["delivered_qty"] < tmp["order_qty"])

    flag = qty_spike | late | cold | backorder
    types = np.where(cold, "COLD_CHAIN",
             np.where(late, "LATE_DELIVERY",
             np.where(qty_spike, "QTY_SPIKE",
             np.where(backorder, "BACKORDER", ""))))

    tmp["anomaly_flag_gen"] = flag.astype(int)
    tmp["anomaly_type_gen"] = types

    current_rate = float(tmp["anomaly_flag_gen"].mean())
    if current_rate > ANOMALY_RATE_TARGET:
        keep_prob = ANOMALY_RATE_TARGET / (current_rate + 1e-9)
        drop_mask = (tmp["anomaly_flag_gen"] == 1) & (rng.random(len(tmp)) > keep_prob)
        tmp.loc[drop_mask, "anomaly_flag_gen"] = 0
        tmp.loc[drop_mask, "anomaly_type_gen"] = ""

    orders_df = orders_df.merge(tmp[["order_id", "anomaly_flag_gen", "anomaly_type_gen"]], on="order_id", how="left")

    return orders_df, shipments_df, inv_df, shock_supplier, shock_days


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    suppliers, pharmacies, products = make_dims(rng)
    orders, shipments, inventory, shock_supplier, shock_days = generate_orders_shipments_inventory(
        suppliers, pharmacies, products, rng
    )

    suppliers.to_csv(os.path.join(OUT_DIR, "dim_supplier.csv"), index=False)
    pharmacies.to_csv(os.path.join(OUT_DIR, "dim_pharmacy.csv"), index=False)
    products.to_csv(os.path.join(OUT_DIR, "dim_product.csv"), index=False)
    orders.to_csv(os.path.join(OUT_DIR, "fact_orders_daily.csv"), index=False)
    shipments.to_csv(os.path.join(OUT_DIR, "fact_shipments.csv"), index=False)
    inventory.to_csv(os.path.join(OUT_DIR, "fact_inventory_daily.csv"), index=False)

    print("[SAVED]", OUT_DIR)
    print("orders:", orders.shape, "shipments:", shipments.shape, "inventory:", inventory.shape)
    print("generated anomaly rate:", float(orders["anomaly_flag_gen"].mean()))
    print("shock supplier:", shock_supplier, "| shock days:", len(shock_days))


if __name__ == "__main__":
    main()

