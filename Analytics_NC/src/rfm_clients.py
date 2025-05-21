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
    # Function responsible to load user's data

    # Reset the file pointer to the beginning
    file_path.seek(0)

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Convert 'Sales_Date' to datetime
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')

    # Drop rows with missing 'Sales_Date'
    data.dropna(subset=['Sales_Date'], inplace=True)

    # Calculate recency in days
    data['Recency'] = (datetime.now() - data['Sales_Date']).dt.days

    # Return the preprocessed DataFrame
    return data

def rfm_clients_compute_rfm(data):
    # Function that computes RFM values from the input data

    # Get the current date
    current_date = data['Sales_Date'].max() + pd.Timedelta(days=1)

    # Group data by 'Client_ID' and aggregate RFM metrics
    rfm_table = data.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Calculate recency
        'Client_ID': 'count',  # Count frequency
        'Monetary_Value': 'sum'  # Sum monetary value
    }).rename(columns={
        'Sales_Date': 'Recency',
        'Client_ID': 'Frequency',
        'Monetary_Value': 'Monetary'
    }).reset_index()  # Reset index

    # Return the RFM table
    return rfm_table

def rfm_clients_normalize_rfm(rfm_table):
    # Normalizes RFM values

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Normalize 'Recency', 'Frequency', and 'Monetary' columns
    rfm_table[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])

    # Calculate RFM score as the mean of normalized RFM values and scale it to 0-100 range
    rfm_table['RFM_Score'] = rfm_table[['Recency', 'Frequency', 'Monetary']].mean(axis=1) * 100

    # Return the normalized RFM table
    return rfm_table

def rfm_clients_perform_clustering(rfm_table, n_clusters=8):
    # This function performs clustering on RFM scores

    # Initialize KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Assign clusters based on RFM scores
    rfm_table['Cluster'] = kmeans.fit_predict(rfm_table[['RFM_Score']].values.reshape(-1, 1))

    # Transform RFM scores to confidence intervals
    rfm_table['Confidence_Interval'] = kmeans.transform(rfm_table[['RFM_Score']].values.reshape(-1, 1)).min(axis=1)

    # Normalize confidence intervals to 0-100 range and clip to a maximum of 100
    rfm_table['Confidence_Interval'] = (1 - MinMaxScaler().fit_transform(rfm_table[['Confidence_Interval']])) * 100
    rfm_table['Confidence_Interval'] = rfm_table['Confidence_Interval'].clip(upper=100)

    # Define cluster labels
    cluster_labels = [
        'Loyal-client', 'Gold-client', 'VIP-client', 'Promising Customer',
        'Cooling-down-client', 'Hibernating', 'Client-at-risk', 'New-client'
    ]

    # Map cluster labels to cluster indices
    rfm_table['Cluster'] = rfm_table['Cluster'].map({i: cluster_labels[i] for i in range(n_clusters)})

    # Return the clustered RFM table
    return rfm_table

def rfm_clients_merge_client_info(data, rfm_table):
    # Function that merges client info with RFM table
    # Merge client info with RFM table on 'Client_ID' and 'Product_ID'
    client_clusters = data[['Client_ID', 'Product_ID']].drop_duplicates().merge(rfm_table, on='Client_ID')

    # Return merged DataFrame
    return client_clusters[['Client_ID', 'RFM_Score', 'Cluster', 'Confidence_Interval']]

def rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences):
    # Function that plots a tree chart of RFM clusters

    # Get cluster counts as sizes
    sizes = cluster_counts.values
    # Define labels with cluster info
    labels = [f"{label}\n{count} clients\nAvg RFM: {mean_rfm_scores[label]:.2f}\nConfidence: {confidences[label]:.2f}%"
              for label, count in cluster_counts.items()]
    # Assign colors based on cluster labels
    color_labels = [colors[label.split("\n")[0]] for label in labels]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot a tree chart using squarify
    squarify.plot(sizes=sizes, label=labels, color=color_labels, alpha=0.8, ax=ax)

    # Turn off axis
    ax.axis('off')

    # Set title
    plt.title('Client Distribution by RFM Cluster (Tree Chart)')

    return fig  # Return the figure

def rfm_clients_save_table_to_file(result_table, file_name="Client_RFM_table.txt"):
    # Define a function named "rfm_clients_save_table_to_file" that saves result table to a file
    # Generate a string representation of the result table using tabulate
    table_str = tabulate(result_table, headers='keys', tablefmt='grid', showindex=False)
    # Write the table string to the specified file
    with open(file_name, 'w') as f:
        f.write(table_str)
