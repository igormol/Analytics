# rfm_clients.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate
import squarify
import streamlit as st

def rfm_clients_load_data(file_path):
    # Reset the file pointer to the beginning of the file.
    file_path.seek(0)

    # Read the CSV file into a DataFrame.
    data = pd.read_csv(file_path)

    # Convert the 'Sales_Date' column to datetime format, handling errors by setting them as NaT.
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')

    # Drop rows with missing 'Sales_Date' values.
    data.dropna(subset=['Sales_Date'], inplace=True)

    # Calculate the recency for each transaction as the number of days since the last sale.
    data['Recency'] = (datetime.now() - data['Sales_Date']).dt.days

    # Return the preprocessed DataFrame.
    return data

def rfm_clients_compute_rfm(data):
    # Get the current date.
    current_date = data['Sales_Date'].max() + pd.Timedelta(days=1)

    # Compute Recency, Frequency, and Monetary (RFM) metrics for each client.
    rfm_table = data.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Recency
        'Client_ID': 'count',                                  # Frequency
        'Monetary_Value': 'sum'                                # Monetary
    }).rename(columns={
        'Sales_Date': 'Recency',
        'Client_ID': 'Frequency',
        'Monetary_Value': 'Monetary'
    }).reset_index()

    # Return the RFM table.
    return rfm_table

def rfm_clients_normalize_rfm(rfm_table):
    # Initialize a MinMaxScaler to normalize Recency, Frequency, and Monetary values.
    scaler = MinMaxScaler()

    # Normalize Recency, Frequency, and Monetary columns using Min-Max scaling.
    rfm_table[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])

    # Calculate the RFM score as the mean of normalized Recency, Frequency, and Monetary values, scaled to 100.
    rfm_table['RFM_Score'] = rfm_table[['Recency', 'Frequency', 'Monetary']].mean(axis=1) * 100

    # Return the normalized RFM table.
    return rfm_table

def rfm_clients_perform_clustering(rfm_table, n_clusters=8):
    # Initialize a KMeans clustering model with the specified number of clusters.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the KMeans model to the RFM scores and assign cluster labels to each client.
    rfm_table['Cluster'] = kmeans.fit_predict(rfm_table[['RFM_Score']].values.reshape(-1, 1))

    # Compute the confidence interval for each client by taking the minimum distance to cluster centers.
    rfm_table['Confidence_Interval'] = kmeans.transform(rfm_table[['RFM_Score']].values.reshape(-1, 1)).min(axis=1)

    # Normalize the confidence interval to a 0-100 range and clip it to a maximum of 100.
    rfm_table['Confidence_Interval'] = (1 - MinMaxScaler().fit_transform(rfm_table[['Confidence_Interval']])) * 100
    rfm_table['Confidence_Interval'] = rfm_table['Confidence_Interval'].clip(upper=100)

    # Define cluster labels based on cluster indices.
    cluster_labels = [
        'Loyal-client', 'Gold-client', 'VIP-client', 'Promising Customer',
        'Cooling-down-client', 'Hibernating', 'Client-at-risk', 'New-client'
    ]

    # Map cluster indices to cluster labels.
    rfm_table['Cluster'] = rfm_table['Cluster'].map({i: cluster_labels[i] for i in range(n_clusters)})

    # Return the RFM table with cluster labels and confidence intervals.
    return rfm_table

def rfm_clients_merge_client_info(data, rfm_table):
    # Merge the RFM analysis results with client information based on the 'Client_ID' column.
    client_clusters = data[['Client_ID', 'Product_ID']].drop_duplicates().merge(rfm_table, on='Client_ID')
    # Return a DataFrame containing only relevant columns: Client_ID, RFM_Score, Cluster, and Confidence_Interval.
    return client_clusters[['Client_ID', 'RFM_Score', 'Cluster', 'Confidence_Interval']]

def rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences):
    # Extract cluster counts, mean RFM scores, and confidence intervals.
    sizes = cluster_counts.values
    labels = [f"{label}\n{count} clients\nAvg RFM: {mean_rfm_scores[label]:.2f}\nConfidence: {confidences[label]:.2f}%"
              for label, count in cluster_counts.items()]
    color_labels = [colors[label.split("\n")[0]] for label in labels]  # Assign colors to cluster labels.

    # Create a Tree Chart using squarify.
    fig, ax = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=color_labels, alpha=0.8, ax=ax)
    ax.axis('off')
    plt.title('Client Distribution by RFM Cluster (Tree Chart)')

    # Return the Tree Chart figure.
    return fig

def rfm_clients_save_table_to_file(result_table, file_name="Client_RFM_table.txt"):
    # Convert the result table to a formatted string representation.
    table_str = tabulate(result_table, headers='keys', tablefmt='grid', showindex=False)

    # Write the formatted table string to a text file.
    with open(file_name, 'w') as f:
        f.write(table_str)
