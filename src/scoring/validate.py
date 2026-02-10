import pandas as pd

def validate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    خروجی: همان orders + ستون‌های validation_errors و is_valid
    validation_errors: لیست کدهای خطا (string با | جدا شده)
    """
    df = orders.copy()

    errors = []

    # Rule 1: order_qty >= 1
    e_qty = df["order_qty"].fillna(0) < 1
    errors.append(("NEG_OR_ZERO_QTY", e_qty))

    # Rule 2: unit_price > 0
    e_price = df["unit_price"].fillna(0) <= 0
    errors.append(("NON_POSITIVE_PRICE", e_price))

    # Rule 3: requested_delivery_date >= date
    d1 = pd.to_datetime(df["date"], errors="coerce")
    d2 = pd.to_datetime(df["requested_delivery_date"], errors="coerce")
    e_date = (d1.isna()) | (d2.isna()) | (d2 < d1)
    errors.append(("INVALID_REQUESTED_DATE", e_date))

    # Rule 4: missing IDs
    for col in ["order_id", "pharmacy_id", "supplier_id", "product_id"]:
        e = df[col].isna() | (df[col].astype(str).str.len() == 0)
        errors.append((f"MISSING_{col.upper()}", e))

    # جمع‌بندی
    df["validation_errors"] = ""
    for code, mask in errors:
        df.loc[mask, "validation_errors"] = df.loc[mask, "validation_errors"].apply(
            lambda s: (s + "|" if s else "") + code
        )

    df["is_valid"] = df["validation_errors"].eq("")
    return df
