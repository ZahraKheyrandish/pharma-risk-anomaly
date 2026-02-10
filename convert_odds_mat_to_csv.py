import os, glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = "benchmarks/tabular/data/raw"
OUT_DIR = "benchmarks/tabular/data/csv"

X_KEYS = ["X", "x", "data", "Data"]
Y_KEYS = ["y", "Y", "label", "Label", "labels", "gt", "target"]

def normalize_label(y):
    y = np.array(y).reshape(-1)
    uniq = np.unique(y)

    if set(uniq.tolist()) == {0, 1}:
        return y.astype(int)
    if set(uniq.tolist()) == {-1, 1}:
        return (y == -1).astype(int)

    if len(uniq) == 2:
        normal_val = min(uniq)
        return (y != normal_val).astype(int)

    raise ValueError(f"Label is not binary. Unique labels: {uniq}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mats = sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))
    if not mats:
        raise SystemExit(f"No .mat files found in {RAW_DIR}")

    for path in mats:
        base = os.path.splitext(os.path.basename(path))[0]
        mat = loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]

        X = None
        y = None

        for k in X_KEYS:
            if k in mat:
                X = np.array(mat[k])
                break
        for k in Y_KEYS:
            if k in mat:
                y = np.array(mat[k])
                break

        if X is None or y is None:
            arrays = {k: np.array(mat[k]) for k in keys}
            two_d = [(k,v) for k,v in arrays.items() if v.ndim == 2 and min(v.shape) > 1]
            one_d = [(k,v) for k,v in arrays.items() if v.ndim in (1,2) and v.size == max(v.shape)]
            if X is None and two_d:
                X = two_d[0][1]
            if y is None and one_d:
                y = one_d[0][1]

        if X is None or y is None:
            raise ValueError(f"{base}: Could not extract X/y. Keys found: {keys}")

        if X.ndim == 2 and X.shape[0] < X.shape[1] and X.shape[0] < 50:
            X = X.T

        y = normalize_label(y)
        X = X.astype(float)

        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
        df["label"] = y.astype(int)

        out_path = os.path.join(OUT_DIR, f"{base}.csv")
        df.to_csv(out_path, index=False)

        print(f"[OK] {base}: n={len(df)}, d={X.shape[1]}, anomaly_rate={df['label'].mean():.4f}")

if __name__ == "__main__":
    main()