import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv(r"D:\CustomerSegmentation\dataset\customers.csv")

# -------------------------------------------------
# CLEAN COLUMN NAMES
# -------------------------------------------------
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "")
    .str.replace("(k$)", "", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("-", "_")
)

print(df.columns.tolist())
rename_map = {
    "customerid": "customer_id",
    "annualincome": "annual_income",
    "spendingscore1100": "spending_score"
}

df = df.rename(columns=rename_map)

print(df.columns.tolist())

print("Columns:", df.columns.tolist())

# -------------------------------------------------
# REQUIRED COLUMNS
# -------------------------------------------------
required_cols = ["age", "annual_income", "spending_score"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.dropna(subset=required_cols).copy()

# -------------------------------------------------
# FEATURE SELECTION
# -------------------------------------------------
X = df[["age", "annual_income", "spending_score"]]

# -------------------------------------------------
# SCALING
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# ELBOW METHOD
# -------------------------------------------------
os.makedirs("static", exist_ok=True)

wcss = []
for k in range(1, 11):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig("static/elbow_method.png")
plt.close()

# -------------------------------------------------
# FINAL MODEL
# -------------------------------------------------
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

# -------------------------------------------------
# CLUSTER CENTERS IN ORIGINAL SCALE
# -------------------------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(
    centers,
    columns=["age", "annual_income", "spending_score"]
)

print("\nCluster Centers:")
print(centers_df)

# -------------------------------------------------
# SMART CLUSTER LABELING
# -------------------------------------------------
# Rule-based assignment based on income/spending behavior
remaining = set(range(n_clusters))

# budget = lowest income + spending
budget_idx = (centers_df["annual_income"] + centers_df["spending_score"]).idxmin()
remaining.remove(budget_idx)

# premium = highest income + spending
premium_idx = (centers_df["annual_income"] + centers_df["spending_score"]).idxmax()
remaining.remove(premium_idx)

# careful = high income, low spending
careful_idx = max(
    remaining,
    key=lambda i: centers_df.loc[i, "annual_income"] - centers_df.loc[i, "spending_score"]
)
remaining.remove(careful_idx)

# target = high spending, lower income
target_idx = max(
    remaining,
    key=lambda i: centers_df.loc[i, "spending_score"] - centers_df.loc[i, "annual_income"]
)
remaining.remove(target_idx)

# standard = whatever remains
standard_idx = remaining.pop()

cluster_labels = {
    budget_idx: "Budget Customers",
    premium_idx: "Premium Customers",
    careful_idx: "Careful Customers",
    target_idx: "Target Customers",
    standard_idx: "Standard Customers"
}

cluster_recommendations = {
    "Budget Customers": "Offer discounts, coupons, and affordable product bundles.",
    "Premium Customers": "Provide VIP offers, loyalty benefits, and premium service.",
    "Careful Customers": "Focus on trust-building and long-term value offers.",
    "Target Customers": "Use targeted marketing and personalized campaigns.",
    "Standard Customers": "Use regular engagement and cross-sell recommendations."
}

df["customer_segment"] = df["cluster"].map(cluster_labels)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
sil_score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {sil_score:.4f}")

# -------------------------------------------------
# SAVE SEGMENTED DATA
# -------------------------------------------------
df.to_csv("segmented_customers.csv", index=False)

# -------------------------------------------------
# SAVE MODEL, SCALER, AND META
# -------------------------------------------------
pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

segment_meta = {
    "feature_cols": ["age", "annual_income", "spending_score"],
    "cluster_labels": cluster_labels,
    "cluster_recommendations": cluster_recommendations,
    "cluster_centers": centers_df.to_dict(orient="index"),
    "silhouette_score": float(sil_score)
}

pickle.dump(segment_meta, open("segment_meta.pkl", "wb"))

# -------------------------------------------------
# SAVE CLUSTER VISUALIZATION
# -------------------------------------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X["annual_income"],
    X["spending_score"],
    c=clusters,
    cmap="viridis",
    s=30,
    alpha=0.8
)

plt.scatter(
    centers_df["annual_income"],
    centers_df["spending_score"],
    c="red",
    s=250,
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.tight_layout()
plt.savefig("static/customer_clusters.png")
plt.close()

print("\nModel saved successfully as kmeans_model.pkl")
print("Scaler saved successfully as scaler.pkl")
print("Meta saved successfully as segment_meta.pkl")
print("Segmented data saved as segmented_customers.csv")