from flask import Flask, render_template, request
import pickle
import numpy as np
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# -----------------------------
# LOAD TRAINED MODEL + SCALER + META
# -----------------------------
model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
segment_meta = pickle.load(open("segment_meta.pkl", "rb"))

feature_cols = segment_meta["feature_cols"]
cluster_labels = segment_meta["cluster_labels"]
cluster_recommendations = segment_meta["cluster_recommendations"]

# -----------------------------
# MYSQL CONNECTION FUNCTION
# -----------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="customer_segmentation"
    )

# -----------------------------
# INIT DB
# -----------------------------
def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predicted_customers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                age INT,
                annual_income INT,
                spending_score INT,
                customer_segment VARCHAR(100),
                recommendation TEXT
            )
        """)

        # Add recommendation column if table already existed
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'customer_segmentation'
              AND TABLE_NAME = 'predicted_customers'
              AND COLUMN_NAME = 'recommendation'
        """)
        if cursor.fetchone()[0] == 0:
            cursor.execute("ALTER TABLE predicted_customers ADD COLUMN recommendation TEXT")

        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print("Database initialization error:", e)

init_db()

# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def home():
    return render_template("index.html")

# -----------------------------
# PREDICTION
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form.get('age', 0))
        income = int(request.form.get('income', 0))
        score = int(request.form.get('score', 0))

        # IMPORTANT: must match training feature order
        input_data = np.array([[age, income, score]])
        scaled_data = scaler.transform(input_data)

        cluster = int(model.predict(scaled_data)[0])

        segment = cluster_labels.get(cluster, f"Cluster {cluster}")
        recommendation = cluster_recommendations.get(segment, "No recommendation available.")

        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            INSERT INTO predicted_customers
            (age, annual_income, spending_score, customer_segment, recommendation)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (age, income, score, segment, recommendation)

        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()

        return render_template(
            "index.html",
            prediction=segment,
            recommendation=recommendation
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            recommendation=None
        )

# -----------------------------
# DASHBOARD
# -----------------------------
@app.route('/dashboard')
def dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predicted_customers")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT customer_segment, COUNT(*)
            FROM predicted_customers
            GROUP BY customer_segment
            ORDER BY COUNT(*) DESC
        """)
        segment_data = cursor.fetchall()

        chart_labels = [row[0] for row in segment_data]
        chart_values = [row[1] for row in segment_data]

        cursor.execute("""
            SELECT age, annual_income, spending_score, customer_segment, recommendation
            FROM predicted_customers
            ORDER BY id DESC
            LIMIT 5
        """)
        recent_data = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template(
            "dashboard.html",
            total_predictions=total_predictions,
            labels=chart_labels,
            values=chart_values,
            recent_data=recent_data
        )

    except Exception as e:
        return render_template(
            "dashboard.html",
            total_predictions=0,
            labels=[],
            values=[],
            recent_data=[],
            error=str(e)
        )

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)