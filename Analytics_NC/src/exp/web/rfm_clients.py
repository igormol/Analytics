import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate
import squarify
import streamlit as st

def rfm_clients_load_data(file_path):
    data = pd.read_csv(file_path)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    data.dropna(subset=['Sales_Date'], inplace=True)
    return data

def rfm_clients_compute_rfm(data):
    current_date = data['Sales_Date'].max() + pd.Timedelta(days=1)
    rfm_table = data.groupby('Client_ID').agg({
        'Sales_Date': lambda x: (current_date - x.max()).days,
        'Client_ID': 'count',
        'Monetary_Value': 'sum'
    }).rename(columns={
        'Sales_Date': 'Recency',
        'Client_ID': 'Frequency',
        'Monetary_Value': 'Monetary'
    }).reset_index()
    return rfm_table

def rfm_clients_normalize_rfm(rfm_table):
    scaler = MinMaxScaler()
    rfm_table[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])
    rfm_table['RFM_Score'] = rfm_table[['Recency', 'Frequency', 'Monetary']].mean(axis=1) * 100
    return rfm_table

def rfm_clients_perform_clustering(rfm_table, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_table['Cluster'] = kmeans.fit_predict(rfm_table[['RFM_Score']].values.reshape(-1, 1))
    rfm_table['Confidence_Interval'] = kmeans.transform(rfm_table[['RFM_Score']].values.reshape(-1, 1)).min(axis=1)

    # Normalize the Confidence Interval to 0-100 range and clip to a maximum of 100
    rfm_table['Confidence_Interval'] = (1 - MinMaxScaler().fit_transform(rfm_table[['Confidence_Interval']])) * 100
    rfm_table['Confidence_Interval'] = rfm_table['Confidence_Interval'].clip(upper=100)

    cluster_labels = [
        'Loyal-client', 'Gold-client', 'VIP-client', 'Promising Customer',
        'Cooling-down-client', 'Hibernating', 'Client-at-risk', 'New-client'
    ]
    rfm_table['Cluster'] = rfm_table['Cluster'].map({i: cluster_labels[i] for i in range(n_clusters)})
    return rfm_table

def rfm_clients_merge_client_info(data, rfm_table):
    client_clusters = data[['Client_ID', 'Product_ID']].drop_duplicates().merge(rfm_table, on='Client_ID')
    return client_clusters[['Client_ID', 'RFM_Score', 'Cluster', 'Confidence_Interval']]

def rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences, file_name="Client_RFM_tree_chart.png"):
    sizes = cluster_counts.values
    labels = [f"{label}\n{count} clients\nAvg RFM: {mean_rfm_scores[label]:.2f}\nConfidence: {confidences[label]:.2f}%"
              for label, count in cluster_counts.items()]
    color_labels = [colors[label.split("\n")[0]] for label in labels]

    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, color=color_labels, alpha=0.8)
    plt.axis('off')
    plt.title('Client Distribution by RFM Cluster (Tree Chart)')
    plt.savefig(file_name)
    plt.close()

def rfm_clients_save_table_to_file(result_table, file_name="Client_RFM_table.txt"):
    result_table = result_table.sort_values(by='RFM_Score', ascending=False)  # Sort by RFM_Score in descending order
    result_table = result_table.drop_duplicates(subset='Client_ID')  # Keep only the first occurrence of each client ID
    table_str = tabulate(result_table, headers='keys', tablefmt='grid', showindex=False)
    with open(file_name, 'w') as f:
        f.write(table_str)

def run_rfm_clients(data):
    rfm_table = rfm_clients_compute_rfm(data)
    rfm_table = rfm_clients_normalize_rfm(rfm_table)
    rfm_table = rfm_clients_perform_clustering(rfm_table)
    result_table = rfm_clients_merge_client_info(data, rfm_table)

    cluster_counts = result_table['Cluster'].value_counts()
    mean_rfm_scores = result_table.groupby('Cluster')['RFM_Score'].mean()
    confidences = result_table.groupby('Cluster')['Confidence_Interval'].mean()

    colors = {
        'Loyal-client': '#FFD700',    # Gold
        'Gold-client': '#FFA500',     # Orange
        'VIP-client': '#FF4500',      # OrangeRed
        'Promising Customer': '#ADFF2F', # GreenYellow
        'Cooling-down-client': '#00CED1', # DarkTurquoise
        'Hibernating': '#4682B4',    # SteelBlue
        'Client-at-risk': '#FF6347', # Tomato
        'New-client': '#32CD32'      # LimeGreen
    }

    st.write("RFM analysis completed.")

    # Display Tree Chart
    rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)
    st.image('Client_RFM_tree_chart.png', use_column_width=True)

    # Provide Download Option for Tree Chart
    with open("Client_RFM_tree_chart.png", "rb") as file:
        st.download_button(
            label="Download Tree Chart as PNG",
            data=file,
            file_name="Client_RFM_tree_chart.png",
            mime="image/png"
        )

    # Display Result Table
    st.subheader('Client RFM Table')
    st.dataframe(result_table)

    # Provide Download Option for Result Table
    csv = result_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Table as CSV",
        data=csv,
        file_name='Client_RFM_table.csv',
        mime='text/csv'
    )

    # Save Table to File
    rfm_clients_save_table_to_file(result_table)

def other_functionality():
    st.title('Other Functionality')
    st.write("This is a placeholder for other functionality.")

def main():
    st.title('RFM Analysis Web App')

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = rfm_clients_load_data(uploaded_file)

        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio("Choose the app mode", ["RFM Analysis", "Other Functionality"])

        if app_mode == "RFM Analysis":
            run_rfm_clients(data)
        elif app_mode == "Other Functionality":
            other_functionality()

if __name__ == "__main__":
    main()
