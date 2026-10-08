import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sources/data.csv", header=None)

os.makedirs("plots", exist_ok=True)

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

labels = df.iloc[:, 1].values
features = df.iloc[:, 2:].values

counts = pd.Series(labels).value_counts()
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, color=["royalblue", "crimson"])
plt.title("Distribution of Diagnosis (M vs B)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
for i, v in enumerate(counts.values):
    plt.text(i, v + 5, f"{v} ({v / len(labels) * 100:.1f}%)", ha="center")
plt.tight_layout()
plt.savefig("plots/eda_diagnosis_distribution.png", dpi=200)
plt.close()

plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    b_vals = features[labels == "B", i]
    m_vals = features[labels == "M", i]
    plt.hist(b_vals, bins=25, alpha=0.6, label="Benign (B)", color="royalblue", density=True)
    plt.hist(m_vals, bins=25, alpha=0.6, label="Malignant (M)", color="crimson", density=True)
    plt.title(feature_names[i])
    plt.legend()
plt.tight_layout()
plt.savefig("plots/eda_features_histogram.png", dpi=200)
plt.close()

corr = np.corrcoef(features[:, :10], rowvar=False)
plt.figure(figsize=(9, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(10), feature_names[:10], rotation=45, ha="right")
plt.yticks(range(10), feature_names[:10])
for i in range(10):
    for j in range(10):
        plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=8)
plt.title("Correlation Matrix (First 10 Features)")
plt.tight_layout()
plt.savefig("plots/eda_correlation_matrix.png", dpi=200)
plt.close()

stats_df = pd.DataFrame(features, columns=feature_names)
desc = stats_df.describe().T
desc[["mean", "std", "min", "50%", "max"]].to_csv("plots/features_statistics.csv")
