import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, davies_bouldin_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import squarify
import streamlit as st
from io import BytesIO

# Load and preprocess data
def load_and_preprocess_data(data):
    data['Brand_Name'].fillna('Unknown', inplace=True)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'])
    today = datetime.now()
    data['Recency'] = (today - data['Sales_Date']).dt.days
    rfm = data.groupby('Client_ID').agg({
        'Recency': 'min',
        'Sales_Date': 'count',  # Frequency
        'Monetary_Value': 'sum'  # Monetary
    }).rename(columns={'Sales_Date': 'Frequency'}).reset_index()
    return rfm

# Calculate churn index
def calculate_churn_index(rfm):
    def churn_index(row):
        return row['Recency'] * (1 / (row['Frequency'] + 1)) * (1 / (row['Monetary_Value'] + 1))
    rfm['Churn_Index'] = rfm.apply(churn_index, axis=1)
    rfm['Retention_Score'] = rfm['Churn_Index'].round(2)
    return rfm

# Normalize features
def normalize_features(rfm):
    scaler = MinMaxScaler(feature_range=(0, 100))
    rfm[['Monetary_Value', 'Churn_Index']] = scaler.fit_transform(rfm[['Monetary_Value', 'Churn_Index']])
    return rfm

# Deep Clustering Network (DCN)
class DCN(models.Model):
    def __init__(self, input_dim, latent_dim):
        super(DCN, self).__init__()
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation=None)
        ])

    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed

# Train DCN
def train_dcn(features, input_dim, latent_dim=3, epochs=10, batch_size=256):
    dcn = DCN(input_dim, latent_dim)
    dcn.compile(optimizer='adam', loss='mse')
    dcn.fit(features, features, epochs=epochs, batch_size=batch_size, verbose=1)
    latent_features = dcn.encoder.predict(features)
    return latent_features

# Perform clustering
def perform_clustering(latent_features, rfm, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(latent_features)
    rfm['Cluster'] = kmeans.labels_
    return rfm, kmeans

# Define cluster names
def define_cluster_names(rfm):
    cluster_summary = rfm.groupby('Cluster').agg({
        'Monetary_Value': 'mean',
        'Churn_Index': 'mean'
    }).reset_index()

    # Sort clusters by Monetary_Value and Churn_Index
    cluster_summary = cluster_summary.sort_values(by=['Monetary_Value', 'Churn_Index'], ascending=[False, True])

    cluster_names = {}
    cluster_names[cluster_summary.iloc[0]['Cluster']] = "High-Value Active"
    cluster_names[cluster_summary.iloc[1]['Cluster']] = "High-Value At-Risk"
    cluster_names[cluster_summary.iloc[2]['Cluster']] = "Low-Value Active"
    cluster_names[cluster_summary.iloc[3]['Cluster']] = "Low-Value At-Risk"

    rfm['Cluster'] = rfm['Cluster'].map(cluster_names)
    return rfm

# Plot treemap
def plot_treemap(rfm, mean_silhouette_scores):
    cluster_stats = rfm.groupby('Cluster').agg({
        'Client_ID': 'count',
        'Monetary_Value': 'mean',
        'Churn_Index': 'mean'
    }).rename(columns={'Client_ID': 'Count'}).reset_index()

    cluster_stats['Mean_Silhouette_Score'] = mean_silhouette_scores
    cluster_stats['Percentage'] = (cluster_stats['Count'] / cluster_stats['Count'].sum()) * 100
    cluster_stats['Label'] = cluster_stats.apply(lambda row: f"{row['Cluster']}\n{row['Percentage']:.2f}%\nM: {row['Monetary_Value']:.2f}\nChurn: {row['Churn_Index']:.2f}\nSilhouette: {row['Mean_Silhouette_Score']:.2f}", axis=1)

    sizes = cluster_stats['Percentage'].tolist()
    labels = cluster_stats['Label'].tolist()
    colors = ["#D1E8E2", "#19747E", "#A9D6E5", "#E2E2E2"]

    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, alpha=0.8, color=colors)
    plt.axis('off')
    plt.title('Client Clustering Treemap')
    return plt

# Save treemap to PNG
def save_treemap_to_png(rfm, mean_silhouette_scores):
    fig = plot_treemap(rfm, mean_silhouette_scores)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Compute Silhouette Scores
def compute_silhouette_scores(rfm, latent_features, kmeans):
    silhouette_vals = silhouette_samples(latent_features, kmeans.labels_)
    rfm['Silhouette_Score'] = silhouette_vals
    mean_silhouette_scores = rfm.groupby('Cluster')['Silhouette_Score'].mean().values
    return rfm, mean_silhouette_scores

# Compute Davies-Bouldin Index
def compute_davies_bouldin_index(latent_features, kmeans):
    db_index = davies_bouldin_score(latent_features, kmeans.labels_)
    return db_index

# Main function
def main():
    st.title("Client Clustering App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        tab = st.sidebar.radio("Select Tab", ["Client Clustering", "New Functionality"])

        if tab == "Client Clustering":
            rfm = load_and_preprocess_data(data)
            rfm = calculate_churn_index(rfm)
            rfm = normalize_features(rfm)
            features = rfm[['Monetary_Value', 'Churn_Index']].values
            input_dim = features.shape[1]
            latent_features = train_dcn(features, input_dim)
            rfm, kmeans = perform_clustering(latent_features, rfm)
            rfm = define_cluster_names(rfm)

            # Sort rfm by Retention_Score in descending order
            rfm_sorted = rfm.sort_values(by='Retention_Score', ascending=False)

            # Compute Silhouette Scores
            rfm_sorted, mean_silhouette_scores = compute_silhouette_scores(rfm_sorted, latent_features, kmeans)

            # Compute Davies-Bouldin Index
            db_index = compute_davies_bouldin_index(latent_features, kmeans)

            client_clustering_tab(rfm_sorted, mean_silhouette_scores, db_index)
        elif tab == "New Functionality":
            new_functionality_tab(data)

def client_clustering_tab(rfm_sorted, mean_silhouette_scores, db_index):
    tab1, tab2, tab3 = st.tabs(["Treemap", "Table", "Davies-Bouldin Index"])

    with tab1:
        st.header("Treemap Chart")
        treemap_fig = plot_treemap(rfm_sorted, mean_silhouette_scores)
        st.pyplot(treemap_fig)

        treemap_buf = save_treemap_to_png(rfm_sorted, mean_silhouette_scores)
        st.download_button(
            label="Download Treemap as PNG",
            data=treemap_buf,
            file_name="treemap.png",
            mime="image/png"
        )

    with tab2:
        st.header("Clustered Clients Table")

        # Create the download button at the top
        csv = rfm_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name='Clustering_Results.csv',
            mime='text/csv',
        )

        st.table(rfm_sorted[['Client_ID', 'Cluster', 'Monetary_Value', 'Retention_Score', 'Silhouette_Score']])

    with tab3:
        st.header("Davies-Bouldin Index")
        db_data = {
            "Cluster": list(rfm_sorted['Cluster'].unique()),
            "Davies-Bouldin Index": [db_index] * len(rfm_sorted['Cluster'].unique())
        }
        db_df = pd.DataFrame(db_data)
        db_df.loc[len(db_df)] = ["Mean", db_index]
        st.table(db_df)

def new_functionality_tab(data):
    st.header("New Functionality")
    # Add your new functionality here

if __name__ == "__main__":
    main()
