
from sklearn.cluster import KMeans
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load data
data = pd.read_csv("Data_Preparation/dataset(preprocessed).csv")
# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# Train model
kmeans = KMeans(n_clusters=4, random_state=66, n_init=10)
kmeans.fit(data_scaled)
# Save model
joblib.dump(kmeans, "Clustering_Analysis/kmeans_model.pkl")
print("Model saved successfully.")
