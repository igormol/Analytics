# rfm_products.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate
import squarify
import streamlit as st

# Set up matplotlib to work with Streamlit
import matplotlib
matplotlib.use('Agg')

def rfm_products_compute_rfm(data):
    # Find the current date in the dataset
    current_date = data['Sales_Date'].max() + pd.Timedelta(days=1)

    # Aggregate sales data by Product_ID and calculate Recency, Frequency, and Monetary metrics
    rfm_table = data.groupby('Product_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,  # Compute recency
        'Product_ID': 'count',                                  # Count the number of transactions (frequency)
        'Monetary_Value': 'sum'                                 # Calculate the total monetary value
    }).rename(columns={
        'Sales_Date': 'Recency',
        'Product_ID': 'Frequency',
        'Monetary_Value': 'Monetary'
    }).reset_index()

    return rfm_table

def rfm_products_normalize_rfm(rfm_table):
    # Initialize MinMaxScaler for normalization
    scaler = MinMaxScaler()

    # Normalize the 'Recency', 'Frequency', and 'Monetary' columns
    rfm_table[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])

    # Calculate the RFM score as the mean of normalized Recency, Frequency, and Monetary scores
    rfm_table['RFM_Score'] = rfm_table[['Recency', 'Frequency', 'Monetary']].mean(axis=1) * 100

    return rfm_table

def rfm_products_perform_clustering(rfm_table, n_clusters=10):
    # Initialize KMeans clustering with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit KMeans to the RFM score values
    rfm_table['Cluster'] = kmeans.fit_predict(rfm_table[['RFM_Score']].values.reshape(-1, 1))

    # Compute cluster ranks based on cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    cluster_ranks = np.zeros_like(sorted_indices)
    cluster_ranks[sorted_indices] = np.arange(n_clusters)

    # Invert ranks to make 0 the highest
    rfm_table['Cluster_Rank'] = rfm_table['Cluster'].map(lambda x: cluster_ranks[x])
    rfm_table['Cluster_Rank'] = rfm_table['Cluster_Rank'].max() - rfm_table['Cluster_Rank']

    # Drop the 'Cluster' column
    rfm_table.drop(columns=['Cluster'], inplace=True)

    # Compute confidence intervals based on distances to cluster centroids
    rfm_table['Confidence_Interval'] = kmeans.transform(rfm_table[['RFM_Score']].values.reshape(-1, 1)).min(axis=1)
    rfm_table['Confidence_Interval'] = (1 - MinMaxScaler().fit_transform(rfm_table[['Confidence_Interval']])) * 100
    rfm_table['Confidence_Interval'] = rfm_table['Confidence_Interval'].clip(upper=100)

    # Map cluster ranks to descriptive cluster labels
    cluster_labels = {
        0: 'Champions', 1: 'Great Performers', 2: 'Potential Stars', 3: 'Rising Stars',
        4: 'Consistent Revenue', 5: 'New Entrants', 6: 'Needs Attention', 7: 'Low Engagement',
        8: 'At Risk', 9: 'Dormant'
    }
    rfm_table['Cluster'] = rfm_table['Cluster_Rank'].map(cluster_labels)

    # Drop the 'Cluster_Rank' column
    rfm_table.drop(columns=['Cluster_Rank'], inplace=True)

    return rfm_table

def rfm_products_merge_product_info(data, rfm_table):
    # Merge product information with RFM table on Product_ID
    product_clusters = data[['Product_ID', 'Product_Name']].drop_duplicates().merge(rfm_table, on='Product_ID')

    # Select relevant columns
    product_info = product_clusters[['Product_ID', 'Product_Name', 'RFM_Score', 'Cluster', 'Confidence_Interval']]

    return product_info

def rfm_products_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences):
    # Compute sizes for each cluster based on product counts
    sizes = cluster_counts.values
    # Generate labels for each cluster with product count, average RFM score, and confidence level
    labels = [f"{label}\n{count} products\nAvg RFM: {mean_rfm_scores[label]:.2f}\nConfidence: {confidences[label]:.2f}%"
              for label, count in cluster_counts.items()]
    # Assign colors for labels based on cluster names
    color_labels = [colors[label.split("\n")[0]] for label in labels]

    # Create a new figure for the tree chart
    fig = plt.figure(figsize=(12, 8))
    # Plot the tree chart using squarify, with sizes, labels, colors, and alpha
    squarify.plot(sizes=sizes, label=labels, color=color_labels, alpha=0.8)
    # Turn off axis for better visualization
    plt.axis('off')
    # Set title for the tree chart
    plt.title('Product Distribution by RFM Cluster (Tree Chart)')
    # Return the generated figure
    return fig

def rfm_products_plot_pie_chart(cluster_counts, colors):
    # Create a new figure for the pie chart
    fig = plt.figure(figsize=(8, 8))
    # Plot the pie chart with cluster counts, labels, autopct formatting, colors, and start angle
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', colors=[colors[label] for label in cluster_counts.index], startangle=140)
    # Set title for the pie chart
    plt.title('Percentage of Products in Each RFM Cluster')
    # Set equal aspect ratio for a circular pie
    plt.axis('equal')
    # Return the generated figure
    return fig

def rfm_products_display_table(result_table):
    # Display header for the RFM analysis results
    st.write("### RFM Analysis Results")
    # Display the result table using Streamlit
    st.write(result_table)
