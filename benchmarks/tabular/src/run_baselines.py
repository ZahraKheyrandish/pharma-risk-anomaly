import os, glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

CSV_DIR = "benchmarks/tabular/data/csv"
REPORT_DIR = "benchmarks/tabular/reports"

def precision_at_k(y_true, scores, k):
    k = max(1, min(k, len(scores)))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())

def _resolve_contamination(contamination, ytr):
    """
    اگر contamination='auto' باشد، از نرخ ناهنجاری train استفاده می‌کنیم.
    سپس مقدار را در بازه (0.001, 0.5] محدود می‌کنیم تا برای مدل‌ها پایدار باشد.
    """
    if contamination == "auto":
        contamination = float(np.mean(ytr))
    contamination = float(contamination)
    contamination = max(0.001, min(0.5, contamination))
    return contamination

def run_one_dataset(path, contamination="auto", seed=42, test_size=0.3):
    df = pd.read_csv(path)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # ✅ تغییر A: contamination را از train می‌گیریم (همسان با plot_pr_curves)
    contamination = _resolve_contamination(contamination, ytr)

    # k را نزدیک تعداد آنومالی‌ها در تست می‌گیریم (audit scenario)
    k = max(10, int(round(len(yte) * yte.mean())))

    rows = []

    # Isolation Forest
    iforest = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    iforest.fit(Xtr)
    scores = -iforest.score_samples(Xte)
    rows.append((
        "IsolationForest",
        roc_auc_score(yte, scores),
        average_precision_score(yte, scores),
        precision_at_k(yte, scores, k),
        k
    ))

    # LOF (novelty=True)
    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination,
        novelty=True
    )
    lof.fit(Xtr)
    scores = -lof.score_samples(Xte)
    rows.append((
        "LOF",
        roc_auc_score(yte, scores),
        average_precision_score(yte, scores),
        precision_at_k(yte, scores, k),
        k
    ))

    # One-Class SVM
   
    nu = min(0.5, max(0.01, contamination))
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(Xtr)
    scores = -ocsvm.score_samples(Xte)
    rows.append((
        "OneClassSVM",
        roc_auc_score(yte, scores),
        average_precision_score(yte, scores),
        precision_at_k(yte, scores, k),
        k
    ))

    return rows, float(np.mean(yte)), contamination  # contamination نهایی را هم برمی‌گردانیم

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs found in {CSV_DIR}")

    all_rows = []
    for path in csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        rows, anomaly_rate_test, contamination_used = run_one_dataset(path)

        for model, roc, pr, p_at_k, k in rows:
            all_rows.append({
                "dataset": name,
                "model": model,
                "anomaly_rate(test)": anomaly_rate_test,
                "contamination(train_used)": contamination_used,
                "ROC_AUC": roc,
                "PR_AUC": pr,
                "Precision@k": p_at_k,
                "k": k
            })

        print(f"[DONE] {name} (test anomaly rate ≈ {anomaly_rate_test:.4f}, contamination used ≈ {contamination_used:.4f})")

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["dataset","PR_AUC"], ascending=[True, False])
    out_path = os.path.join(REPORT_DIR, "results.csv")
    results.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print(results.groupby("dataset")[["PR_AUC","ROC_AUC"]].max())

if __name__ == "__main__":
    main()
