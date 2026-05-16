# Customer Segmentation System

A Machine Learning-based Customer Segmentation System developed using Flask, Scikit-learn, MySQL, and K-Means Clustering to analyze customer behavior and classify customers into meaningful business segments for targeted marketing and business analytics.

---

# Project Overview

This project identifies different types of customers based on their:

- Age
- Annual Income
- Spending Score

Using K-Means Clustering, customers are grouped into multiple segments such as:

- Premium Customers
- High Value Customers
- Target Customers
- Budget Customers
- Careful Customers

The system also provides:

- Real-time customer segment prediction
- Business recommendations
- Analytics dashboard
- Prediction history storage
- Data visualization

---

# Features

## Customer Segmentation Prediction
Predicts customer category using Machine Learning clustering algorithms.

## Dashboard Analytics
Interactive dashboard showing:

- Total predictions
- Segment distribution
- Recent customer predictions
- Pie chart visualization

## Recommendation Engine
Provides business suggestions based on customer type.

Example:
- Premium Customers → VIP offers and loyalty rewards
- Budget Customers → Discounts and affordable bundles
- Careful Customers → Trust-building strategies

## Database Integration
Stores customer prediction history using MySQL.

## Machine Learning Pipeline
Includes:
- Data preprocessing
- Feature scaling
- K-Means clustering
- Elbow Method analysis
- Silhouette Score evaluation

---

# Technologies Used

## Frontend
- HTML
- CSS
- JavaScript
- Chart.js

## Backend
- Python
- Flask

## Machine Learning
- Scikit-learn
- K-Means Clustering
- StandardScaler

## Database
- MySQL

## Data Processing
- Pandas
- NumPy

## Visualization
- Matplotlib

---

# Machine Learning Workflow

1. Load customer dataset
2. Clean and preprocess data
3. Select important features
4. Apply feature scaling
5. Train K-Means clustering model
6. Evaluate clusters using:
   - Elbow Method
   - Silhouette Score
7. Save trained model and scaler
8. Predict customer segments
9. Store predictions in MySQL
10. Display analytics on dashboard

---

# Customer Segments

| Segment | Description |
|---|---|
| Premium Customers | High spending and high income customers |
| High Value Customers | Loyal customers with good purchasing behavior |
| Target Customers | Potential customers for business growth |
| Budget Customers | Low income and low spending customers |
| Careful Customers | Customers who spend carefully and require trust-building |

---

# Project Structure

bash
CustomerSegmentation/
│
├── static/
│   ├── style.css
│   ├── elbow_method.png
│   ├── customer_clusters.png
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│
├── dataset/
│   └── customers.csv
│
├── app.py
├── train.py
├── kmeans_model.pkl
├── scaler.pkl
├── segment_meta.pkl
├── segmented_customers.csv
├── requirements.txt
└── README.md


---

# Dashboard Features

The dashboard includes:

- Total customer predictions
- Pie chart distribution of customer segments
- Recent predictions table
- Recommendation system
- Business analytics visualization

---

# Model Performance

- Algorithm Used: K-Means Clustering
- Number of Clusters: 5
- Silhouette Score: 0.4166

---

# Screenshots

## Home Page
<img width="969" height="609" alt="image" src="https://github.com/user-attachments/assets/92671355-1711-4f36-9a63-ca5404926f29" />

## Prediction Result
(Add screenshot here)

## Dashboard Analytics
(Add screenshot here)

## Cluster Visualization
(Add screenshot here)

## Elbow Method Graph
(Add screenshot here)

---

# Installation Guide

## Clone Repository

bash
git clone https://github.com/your-username/customer-segmentation-system.git


## Move to Project Folder

bash
cd customer-segmentation-system


## Install Required Libraries

bash
pip install -r requirements.txt


## Setup MySQL Database

Create database:

sql
CREATE DATABASE customer_segmentation;


Update MySQL credentials inside:

python
app.py


## Train Model

bash
python train.py


## Run Flask Application

bash
python app.py


---

# Business Use Cases

- Personalized marketing
- Customer targeting
- Loyalty program optimization
- Product recommendation systems
- Customer retention analysis
- Business intelligence analytics

---

# Future Improvements

- AI-powered customer recommendations
- Streamlit analytics dashboard
- Real-time prediction monitoring
- PCA visualization
- Deep learning-based segmentation
- Customer churn prediction
- Generative AI business insights
- Cloud deployment using AWS/GCP

---

# Learning Outcomes

This project helped in understanding:

- Unsupervised Machine Learning
- Customer behavior analytics
- Business intelligence concepts
- Flask web development
- Database integration
- Data visualization
- Real-world ML deployment

---

# Author

Pruthvi M

GitHub: https://github.com/pruthvimkadri

---

# License

This project is developed for educational and portfolio purposes.
