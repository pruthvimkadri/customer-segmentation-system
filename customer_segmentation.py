# =========================================================
# CUSTOMER SEGMENTATION PROJECT (UPDATED WITH SAVED GRAPHS)
# =========================================================

# =========================================================
# PART 1 - IMPORT LIBRARIES
# =========================================================

import os
import pickle
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# PART 2 - CREATE STATIC FOLDER IF NOT EXISTS
# =========================================================

os.makedirs("static", exist_ok=True)

# =========================================================
# PART 3 - CONNECT MYSQL
# =========================================================

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="customer_segmentation"
)

print("✅ MySQL Connected Successfully")

# =========================================================
# PART 4 - LOAD DATA FROM MYSQL
# =========================================================

query = "SELECT * FROM customers"
df = pd.read_sql(query, conn)

print("\n✅ DATA LOADED SUCCESSFULLY\n")
print(df.head())

# =========================================================
# PART 5 - BASIC EDA
# =========================================================

print("\n==============================")
print("DATA INFORMATION")
print("==============================")
print(df.info())

print("\n==============================")
print("NULL VALUES")
print("==============================")
print(df.isnull().sum())

print("\n==============================")
print("STATISTICS")
print("==============================")
print(df.describe())

# =========================================================
# PART 6 - AGE DISTRIBUTION GRAPH
# =========================================================

plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/age_distribution.png")
plt.show()

# =========================================================
# PART 7 - INCOME VS SPENDING SCORE
# =========================================================

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['annual_income'], y=df['spending_score'])
plt.title("Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig("static/income_vs_spending.png")
plt.show()

# =========================================================
# PART 8 - FEATURE SELECTION
# =========================================================

X = df[['annual_income', 'spending_score']]

# =========================================================
# PART 9 - SCALING
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use if needed
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("✅ Scaler saved")

# =========================================================
# PART 10 - ELBOW METHOD
# =========================================================

wcss = []

for i in range(1, 11):
    kmeans_test = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans_test.fit(X_scaled)
    wcss.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig("static/elbow_method.png")
plt.show()

# =========================================================
# PART 11 - TRAIN KMEANS MODEL
# =========================================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans.fit_predict(X_scaled)

print("\n✅ CLUSTERING COMPLETED")
print(df.groupby('cluster')[['annual_income', 'spending_score']].mean())

# Save trained model
pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
print("✅ KMeans model saved")

# =========================================================
# PART 12 - MEANINGFUL CUSTOMER LABELS
# =========================================================

cluster_names = {
    0: "Careful Customers",
    1: "Standard Customers",
    2: "Premium Customers",
    3: "Budget Customers",
    4: "High Value Customers"
}

df['customer_type'] = df['cluster'].map(cluster_names)

print("\n✅ CUSTOMER LABELS ADDED")
print(df[['annual_income', 'spending_score', 'customer_type']].head())

# =========================================================
# PART 13 - CUSTOMER SEGMENTATION GRAPH
# =========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['annual_income'],
    y=df['spending_score'],
    hue=df['customer_type'],
    s=100
)
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig("static/customer_segments.png")
plt.show()

# =========================================================
# PART 14 - CLUSTER SUMMARY
# =========================================================

cluster_summary = df.groupby('customer_type')[['annual_income', 'spending_score', 'age']].mean()

print("\n==============================")
print("CUSTOMER GROUP SUMMARY")
print("==============================")
print(cluster_summary)

# =========================================================
# PART 15 - CREATE TABLE FOR SEGMENTED CUSTOMERS
# =========================================================

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS segmented_customers (
    customer_id INT,
    gender VARCHAR(10),
    age INT,
    annual_income INT,
    spending_score INT,
    cluster INT,
    customer_type VARCHAR(50)
)
""")

conn.commit()
print("\n✅ segmented_customers TABLE READY")

# =========================================================
# PART 16 - CLEAR OLD DATA
# =========================================================

cursor.execute("DELETE FROM segmented_customers")
conn.commit()
print("✅ OLD DATA CLEARED")

# =========================================================
# PART 17 - INSERT SEGMENTED DATA INTO MYSQL
# =========================================================

for _, row in df.iterrows():
    sql = """
    INSERT INTO segmented_customers
    (
        customer_id,
        gender,
        age,
        annual_income,
        spending_score,
        cluster,
        customer_type
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        int(row['customer_id']),
        row['gender'],
        int(row['age']),
        int(row['annual_income']),
        int(row['spending_score']),
        int(row['cluster']),
        row['customer_type']
    )

    cursor.execute(sql, values)

conn.commit()
print("\n✅ SEGMENTED DATA SAVED INTO MYSQL")

# =========================================================
# PART 18 - SAVE CSV FILE
# =========================================================

df.to_csv("segmented_customers.csv", index=False)
print("\n✅ CSV FILE SAVED")

# =========================================================
# PART 19 - CLOSE MYSQL CONNECTION
# =========================================================

cursor.close()
conn.close()
print("\n✅ MYSQL CONNECTION CLOSED")

# =========================================================
# PROJECT COMPLETED
# =========================================================

print("\n🎉 CUSTOMER SEGMENTATION PROJECT COMPLETED SUCCESSFULLY")