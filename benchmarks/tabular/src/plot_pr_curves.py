import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

CSV_DIR = "benchmarks/tabular/data/csv"
OUT_DIR = "benchmarks/tabular/reports/pr_curves"

def fit_and_scores(Xtr, Xte, contamination=0.05, seed=42):
    # Ensure contamination is a sane float in (0, 0.5]
    contamination = float(contamination)
    contamination = max(0.001, min(0.5, contamination))

    models = {}

    models["IsolationForest"] = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )

    models["LOF"] = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination,
        novelty=True
    )

    nu = min(0.5, max(0.01, contamination))
    models["OneClassSVM"] = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")

    scores = {}
    for name, model in models.items():
        model.fit(Xtr)
        scores[name] = -model.score_samples(Xte)  # higher = more anomalous
    return scores

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs found in {CSV_DIR}")

    for path in csvs:
        ds = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        y = df["label"].astype(int).values
        X = df.drop(columns=["label"]).values

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        # Use the observed anomaly rate in the training split as contamination
        contamination = float(ytr.mean())
        scores = fit_and_scores(Xtr, Xte, contamination=contamination)

        plt.figure()
        for name, s in scores.items():
            p, r, _ = precision_recall_curve(yte, s)
            ap = average_precision_score(yte, s)
            plt.plot(r, p, drawstyle="steps-post", label=f"{name} (AP={ap:.3f})")

        # Random baseline: precision equals prevalence (anomaly rate)
        plt.axhline(yte.mean(), linestyle="--", linewidth=1, label="Random baseline")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve — {ds} (anomaly rate={yte.mean():.3f})")
        plt.legend()
        plt.grid(True, alpha=0.25)

        out_path = os.path.join(OUT_DIR, f"{ds}_pr.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
