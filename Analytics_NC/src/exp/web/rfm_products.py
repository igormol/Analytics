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

# Function to load data
def rfm_products_load_data(file_path):
    data = pd.read_csv(file_path)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    data.dropna(subset=['Sales_Date'], inplace=True)
    return data

# Function to compute RFM
def rfm_products_compute_rfm(data):
    current_date = data['Sales_Date'].max() + pd.Timedelta(days=1)
    rfm_table = data.groupby('Product_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,
        'Product_ID': 'count',
        'Monetary_Value': 'sum'
    }).rename(columns={
        'Sales_Date': 'Recency',
        'Product_ID': 'Frequency',
        'Monetary_Value': 'Monetary'
    }).reset_index()
    return rfm_table

# Function to normalize RFM
def rfm_products_normalize_rfm(rfm_table):
    scaler = MinMaxScaler()
    rfm_table[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])
    rfm_table['RFM_Score'] = rfm_table[['Recency', 'Frequency', 'Monetary']].mean(axis=1) * 100
    return rfm_table

# Function to perform clustering
def rfm_products_perform_clustering(rfm_table, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_table['Cluster'] = kmeans.fit_predict(rfm_table[['RFM_Score']].values.reshape(-1, 1))

    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    cluster_ranks = np.zeros_like(sorted_indices)
    cluster_ranks[sorted_indices] = np.arange(n_clusters)

    rfm_table['Cluster_Rank'] = rfm_table['Cluster'].map(lambda x: cluster_ranks[x])
    rfm_table['Cluster_Rank'] = rfm_table['Cluster_Rank'].max() - rfm_table['Cluster_Rank']  # Invert ranks to make 0 the highest
    rfm_table.drop(columns=['Cluster'], inplace=True)

    rfm_table['Confidence_Interval'] = kmeans.transform(rfm_table[['RFM_Score']].values.reshape(-1, 1)).min(axis=1)
    rfm_table['Confidence_Interval'] = (1 - MinMaxScaler().fit_transform(rfm_table[['Confidence_Interval']])) * 100
    rfm_table['Confidence_Interval'] = rfm_table['Confidence_Interval'].clip(upper=100)

    cluster_labels = [
        'Champions', 'Great Performers', 'Potential Stars', 'Rising Stars',
        'Consistent Revenue', 'New Entrants', 'Needs Attention', 'Low Engagement',
        'At Risk', 'Dormant'
    ]
    cluster_labels = {i: cluster_labels[i] for i in range(n_clusters)}
    rfm_table['Cluster'] = rfm_table['Cluster_Rank'].map(cluster_labels)
    rfm_table.drop(columns=['Cluster_Rank'], inplace=True)

    return rfm_table

# Function to merge product info
def rfm_products_merge_product_info(data, rfm_table):
    product_clusters = data[['Product_ID', 'Product_Name']].drop_duplicates().merge(rfm_table, on='Product_ID')
    return product_clusters[['Product_ID', 'Product_Name', 'RFM_Score', 'Cluster', 'Confidence_Interval']]

# Function to plot tree chart
def rfm_products_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences):
    sizes = cluster_counts.values
    labels = [f"{label}\n{count} products\nAvg RFM: {mean_rfm_scores[label]:.2f}\nConfidence: {confidences[label]:.2f}%"
              for label, count in cluster_counts.items()]
    color_labels = [colors[label.split("\n")[0]] for label in labels]

    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=color_labels, alpha=0.8)
    plt.axis('off')
    plt.title('Product Distribution by RFM Cluster (Tree Chart)')
    st.pyplot(plt)
    plt.close()

# Function to plot pie chart
def rfm_products_plot_pie_chart(cluster_counts, colors):
    plt.figure(figsize=(8, 8))
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', colors=[colors[label] for label in cluster_counts.index], startangle=140)
    plt.title('Percentage of Products in Each RFM Cluster')
    plt.axis('equal')
    st.pyplot(plt)
    plt.close()

# Function to display result table
def rfm_products_display_table(result_table):
    st.write("### RFM Analysis Results")
    st.write(result_table)

# Function to save table to file
def rfm_products_save_table_to_file(result_table, file_name="Product_RFM_table.txt"):
    result_table = result_table.sort_values(by='RFM_Score', ascending=False)
    result_table = result_table.drop_duplicates(subset='Product_ID')
    table_str = tabulate(result_table, headers='keys', tablefmt='grid', showindex=False)
    with open(file_name, 'w') as f:
        f.write(table_str)

# Function to handle RFM Analysis tab with sub-tabs
def run_rfm_products(data):
    rfm_table = rfm_products_compute_rfm(data)
    rfm_table = rfm_products_normalize_rfm(rfm_table)
    rfm_table = rfm_products_perform_clustering(rfm_table)
    result_table = rfm_products_merge_product_info(data, rfm_table)

    colors = {
        'Champions': '#fcc914',          # Light Yellow
        'Great Performers': '#ffb6c1',   # Light Pink
        'Potential Stars': '#fcc914',    # Orange
        'Rising Stars': '#ff9800',       # Yellow
        'Consistent Revenue': '#37a55a', # Green
        'New Entrants': '#d2b48c',       # Light Brown
        'Needs Attention': '#008db9',    # Icy Blue
        'Low Engagement': '#36a0ce',     # Light Blue
        'At Risk': '#ff6961',            # Light Red
        'Dormant': '#b0e0e6'             # Powder Blue
    }

    st.write("## RFM Analysis")

    sub_tab_selection = st.radio("Select View", ["Tree Chart", "Pie Chart", "RFM Table"])

    if sub_tab_selection == "Tree Chart":
        cluster_counts = result_table['Cluster'].value_counts()
        mean_rfm_scores = result_table.groupby('Cluster')['RFM_Score'].mean()
        confidences = result_table.groupby('Cluster')['Confidence_Interval'].mean()
        rfm_products_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)

    elif sub_tab_selection == "Pie Chart":
        cluster_counts = result_table['Cluster'].value_counts()
        rfm_products_plot_pie_chart(cluster_counts, colors)

    elif sub_tab_selection == "RFM Table":
        rfm_products_display_table(result_table)

    # Save the table to file
    rfm_products_save_table_to_file(result_table)

# Function to handle other functionality
def handle_other_functionality():
    st.write("### Other Functionality")
    st.write("This tab will include additional functionalities to be decided later.")

# Main function to run the app
def rfm_products_main():
    st.sidebar.title("RFM Products Analysis")
    st.title("RFM Products Analysis Web App")

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = rfm_products_load_data(uploaded_file)

        tab_selection = st.sidebar.radio("Select Tab", ["RFM Analysis", "Other Functionality"])

        if tab_selection == "RFM Analysis":
            run_rfm_products(data)
        elif tab_selection == "Other Functionality":
            handle_other_functionality()

if __name__ == "__main__":
    rfm_products_main()
