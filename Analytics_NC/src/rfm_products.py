# rfm_products.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import squarify
import streamlit as st

matplotlib.use('Agg')

def rfm_products_compute_rfm(data):
    # Compute RFM metrics for products

    # Get current date
    current_date = data['Sales_Date'].max() + pd.Timedelta(days=1)

    # Group data by 'Product_ID' and aggregate RFM metrics
    rfm_table = data.groupby('Product_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Calculate recency
        'Product_ID': 'count',  # Count frequency
        'Monetary_Value': 'sum'  # Sum monetary value
    }).rename(columns={
        'Sales_Date': 'Recency',
        'Product_ID': 'Frequency',
        'Monetary_Value': 'Monetary'
    }).reset_index()  # Reset index

    # Return the RFM table
    return rfm_table

def rfm_products_normalize_rfm(rfm_table):
    # Normalize RFM metrics for products

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize 'Recency', 'Frequency', and 'Monetary' columns
    rfm_table[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])

    # Calculate RFM score as the mean of normalized RFM values and scale it to 0-100 range
    rfm_table['RFM_Score'] = rfm_table[['Recency', 'Frequency', 'Monetary']].mean(axis=1) * 100

    # Return the normalized RFM table
    return rfm_table

def rfm_products_perform_clustering(rfm_table, n_clusters=10):
    # Perform clustering on product RFM scores

    # Initialize KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Assign clusters based on RFM scores
    rfm_table['Cluster'] = kmeans.fit_predict(rfm_table[['RFM_Score']].values.reshape(-1, 1))

    # Calculate cluster ranks based on cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    cluster_ranks = np.zeros_like(sorted_indices)
    cluster_ranks[sorted_indices] = np.arange(n_clusters)

    # Invert ranks to make 0 the highest
    rfm_table['Cluster_Rank'] = rfm_table['Cluster'].map(lambda x: cluster_ranks[x])
    rfm_table['Cluster_Rank'] = rfm_table['Cluster_Rank'].max() - rfm_table['Cluster_Rank']
    rfm_table.drop(columns=['Cluster'], inplace=True)  # Drop the 'Cluster' column

    # Normalize confidence intervals to 0-100 range and clip to a maximum of 100
    rfm_table['Confidence_Interval'] = kmeans.transform(rfm_table[['RFM_Score']].values.reshape(-1, 1)).min(axis=1)
    rfm_table['Confidence_Interval'] = (1 - MinMaxScaler().fit_transform(rfm_table[['Confidence_Interval']])) * 100
    rfm_table['Confidence_Interval'] = rfm_table['Confidence_Interval'].clip(upper=100)

    # Define cluster labels
    cluster_labels = [
        'Champions', 'Great Performers', 'Potential Stars', 'Rising Stars',
        'Consistent Revenue', 'New Entrants', 'Needs Attention', 'Low Engagement',
        'At Risk', 'Dormant'
    ]
    cluster_labels = {i: cluster_labels[i] for i in range(n_clusters)}

    # Map cluster ranks to cluster labels
    rfm_table['Cluster'] = rfm_table['Cluster_Rank'].map(cluster_labels)
    rfm_table.drop(columns=['Cluster_Rank'], inplace=True)  # Drop the 'Cluster_Rank' column

    # Return the clustered RFM table
    return rfm_table


def rfm_products_merge_product_info(data, rfm_table):
    # Merge product info with RFM table

    product_clusters = data[['Product_ID', 'Product_Name']].drop_duplicates().merge(rfm_table, on='Product_ID')
    return product_clusters[['Product_ID', 'Product_Name', 'RFM_Score', 'Cluster', 'Confidence_Interval']]

def rfm_products_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences):
    # Plot a tree chart of product RFM clusters

    # Get cluster counts as sizes
    sizes = cluster_counts.values

    # Define labels with cluster info
    labels = [f"{label}\n{count} products\nAvg RFM: {mean_rfm_scores[label]:.2f}\nConfidence: {confidences[label]:.2f}%"
              for label, count in cluster_counts.items()]

    # Assign colors based on cluster labels
    color_labels = [colors[label.split("\n")[0]] for label in labels]

    # Create a figure
    fig = plt.figure(figsize=(12, 8))

    # Plot a tree chart using squarify
    squarify.plot(sizes=sizes, label=labels, color=color_labels, alpha=0.8)

    # Turn off axis
    plt.axis('off')

    # Set title
    plt.title('Product Distribution by RFM Cluster (Tree Chart)')

    # Return the figure
    return fig

def rfm_products_plot_pie_chart(cluster_counts, colors):
    # Plot a pie chart of product RFM clusters

    # Create a figure
    fig = plt.figure(figsize=(8, 8))

    # Plot a pie chart
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
            colors=[colors[label] for label in cluster_counts.index], startangle=140)

    # Set title
    plt.title('Percentage of Products in Each RFM Cluster')

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Return the figure
    return fig

def rfm_products_display_table(result_table):
    # Display RFM analysis results as a table

    # Write title
    st.write("### RFM Analysis Results")

    # Write the result table
    st.write(result_table)
