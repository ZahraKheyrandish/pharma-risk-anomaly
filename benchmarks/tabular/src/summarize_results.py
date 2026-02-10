import pandas as pd

RES = "benchmarks/tabular/reports/results.csv"

df = pd.read_csv(RES)

# بهترین مدل برای هر دیتاست بر اساس PR_AUC
best = (
    df.sort_values(["dataset", "PR_AUC"], ascending=[True, False])
      .groupby("dataset")
      .head(1)
      [["dataset","model","anomaly_rate(test)","PR_AUC","ROC_AUC","Precision@k","k"]]
)

# میانگین عملکرد هر مدل روی همه دیتاست‌ها (macro average)
macro = (
    df.groupby("model")[["PR_AUC","ROC_AUC","Precision@k"]]
      .mean()
      .sort_values("PR_AUC", ascending=False)
)

print("\n=== Best per dataset (by PR_AUC) ===")
print(best.to_string(index=False))

print("\n=== Macro mean over datasets ===")
print(macro.to_string())

best.to_csv("benchmarks/tabular/reports/best_models.csv", index=False)
macro.to_csv("benchmarks/tabular/reports/model_macro_mean.csv")
print("\nSaved: benchmarks/tabular/reports/best_models.csv")
print("Saved: benchmarks/tabular/reports/model_macro_mean.csv")
