from flask import Flask, render_template, request
import pickle
import numpy as np
import mysql.connector
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = pickle.load(open("kmeans_model.pkl", "rb"))

# -----------------------------
# MYSQL CONNECTION
# -----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="customer_segmentation"
)

cursor = db.cursor()

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

    age = int(request.form['age'])
    income = int(request.form['income'])
    score = int(request.form['score'])

    # -----------------------------
    # SCALE INPUT DATA
    # -----------------------------
    scaler = StandardScaler()

    sample_data = np.array([
        [15, 39],
        [16, 81],
        [17, 6],
        [18, 77],
        [19, 40],
        [20, 76],
        [25, 15],
        [30, 90],
        [60, 50],
        [100, 95]
    ])

    scaler.fit(sample_data)

    # Input values
    data = np.array([[income, score]])

    # Scale values
    scaled_data = scaler.transform(data)

    # Predict cluster
    cluster = model.predict(scaled_data)[0]

    # -----------------------------
    # CLUSTER LABELS
    # -----------------------------
    labels = {
        0: "Budget Customers",
        1: "High Value Customers",
        2: "Standard Customers",
        3: "Target Customers",
        4: "Careful Customers"
    }

    result = labels.get(cluster, "Unknown")

    # -----------------------------
    # SAVE TO MYSQL
    # -----------------------------
    query = """
    INSERT INTO predicted_customers(
        age,
        annual_income,
        spending_score,
        customer_segment
    )
    VALUES (%s, %s, %s, %s)
    """

    values = (age, income, score, result)

    cursor.execute(query, values)
    db.commit()

    return render_template(
        "index.html",
        prediction=result
    )
@app.route('/dashboard')
def dashboard():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="customer_segmentation"
    )

    cursor = conn.cursor()

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predicted_customers")
    total_predictions = cursor.fetchone()[0]

    # Segment counts
    cursor.execute("""
        SELECT customer_segment, COUNT(*)
        FROM predicted_customers
        GROUP BY customer_segment
    """)

    segment_data = cursor.fetchall()

    labels = []
    values = []

    for row in segment_data:
        labels.append(row[0])
        values.append(row[1])

    # Recent predictions
    cursor.execute("""
        SELECT age, annual_income,
        spending_score, customer_segment
        FROM predicted_customers
        ORDER BY id DESC
        LIMIT 5
    """)

    recent_data = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        labels=labels,
        values=values,
        recent_data=recent_data
    )
# -----------------------------
# RUN APP
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)