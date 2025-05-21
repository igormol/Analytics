import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    data['Recency'] = (datetime.now() - data['Sales_Date']).dt.days
    return data

# Function to compute churn scores
def compute_churn_scores(data):
    frequency = data.groupby('Client_ID')['Sales_Date'].count()
    recency = data.groupby('Client_ID')['Recency'].min()
    churn_data = pd.DataFrame({'Frequency': frequency, 'Recency': recency})
    churn_data['Frequency'] = 1 / (1 + churn_data['Frequency'])
    churn_data['Recency'] = churn_data['Recency'] / churn_data['Recency'].max()
    churn_data['Churn_Score'] = churn_data[['Frequency', 'Recency']].mean(axis=1)
    return churn_data

# Function to apply K-Means clustering and compute confidence scores
def apply_kmeans_clustering(churn_data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(churn_data[['Churn_Score']])
    churn_data['Cluster'] = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Compute confidence scores based on distances to cluster centroids
    distances = np.linalg.norm(churn_data[['Churn_Score']].values - cluster_centers[churn_data['Cluster']], axis=1)
    max_distance = np.max(distances)
    churn_data['Confidence'] = 1 - (distances / max_distance)
    churn_data['Confidence'] = churn_data['Confidence'] * 100  # Normalize to percentage

    cluster_names = {0: 'Safe', 1: 'Low Risk', 2: 'Moderate Risk', 3: 'High Risk'}
    churn_data['Cluster_Name'] = churn_data['Cluster'].map(cluster_names)
    return churn_data, kmeans

# Function to classify clients based on churn index intervals
def classify_clients(churn_data):
    intervals = {
        'Safe': (churn_data[churn_data['Cluster_Name'] == 'Safe']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'Safe']['Churn_Score'].max()),
        'Low Risk': (churn_data[churn_data['Cluster_Name'] == 'Low Risk']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'Low Risk']['Churn_Score'].max()),
        'Moderate Risk': (churn_data[churn_data['Cluster_Name'] == 'Moderate Risk']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'Moderate Risk']['Churn_Score'].max()),
        'High Risk': (churn_data[churn_data['Cluster_Name'] == 'High Risk']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'High Risk']['Churn_Score'].max())
    }
    return intervals

# Function to create and save the pie chart with the specified modifications
def create_and_save_pie_chart(churn_data, kmeans, output_file):
    cluster_counts = churn_data['Cluster_Name'].value_counts()
    average_churn = churn_data.groupby('Cluster_Name')['Churn_Score'].mean()
    average_confidence = churn_data.groupby('Cluster_Name')['Confidence'].mean()
    labels = [f'{name}\n{count} clients\nAvg Churn: {churn:.2f}\nConfidence: {conf:.2f}'
              for name, count, churn, conf in zip(cluster_counts.index, cluster_counts.values, average_churn.values, average_confidence.values)]

    colors = ['#E07A5F', '#3D405B', '#81B29A', '#F2CC8F']  # Color scheme

    plt.figure(figsize=(10, 10))
    plt.pie(cluster_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Client Churn Clusters')

    # Add legend and position it below the plot area to ensure it is fully visible
    plt.legend(['Safe: Low Churn Score', 'Low Risk: Relatively Stable', 'Moderate Risk: Showing Signs of Churn', 'High Risk: High Churn Likelihood'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    plt.savefig(output_file)
    plt.close()

# Function to print and save the result table
def print_and_save_table(churn_data, data, output_file):
    client_info = data[['Client_ID']].drop_duplicates().set_index('Client_ID')
    result = churn_data.merge(client_info, left_index=True, right_index=True)
    result_sorted = result.sort_values(by='Churn_Score', ascending=False)  # Sort in decreasing order of churn score
    table = tabulate(result_sorted[['Churn_Score', 'Confidence', 'Cluster_Name']].reset_index(), headers='keys', tablefmt='grid')
    with open(output_file, 'w') as f:
        f.write(table)

# Function to create and save the bar chart without the legend
def create_and_save_bar_chart(churn_data, output_file):
    cluster_counts = churn_data['Cluster_Name'].value_counts()
    average_churn = churn_data.groupby('Cluster_Name')['Churn_Score'].mean()
    average_confidence = churn_data.groupby('Cluster_Name')['Confidence'].mean()
    intervals = classify_clients(churn_data)

    total_clients = len(churn_data)
    bar_data = {
        'Cluster': [],
        'Count': [],
        'Percentage': [],
        'Avg_Churn': [],
        'Churn_Interval': [],
        'Avg_Confidence': []
    }

    for cluster_name, count in cluster_counts.items():
        percentage = (count / total_clients) * 100
        avg_churn = average_churn[cluster_name]
        interval = intervals[cluster_name]
        avg_confidence = average_confidence[cluster_name]
        bar_data['Cluster'].append(cluster_name)
        bar_data['Count'].append(count)
        bar_data['Percentage'].append(percentage)
        bar_data['Avg_Churn'].append(avg_churn)
        bar_data['Churn_Interval'].append(f'{interval[0]:.2f} - {interval[1]:.2f}')
        bar_data['Avg_Confidence'].append(avg_confidence)

    df_bar = pd.DataFrame(bar_data)
    colors = ['#E07A5F', '#3D405B', '#81B29A', '#F2CC8F']  # Color scheme

    plt.figure(figsize=(14, 8))
    bars = plt.bar(df_bar['Cluster'], df_bar['Percentage'], color=colors)
    for bar, perc, count, avg_churn, interval, avg_conf in zip(bars, df_bar['Percentage'], df_bar['Count'], df_bar['Avg_Churn'], df_bar['Churn_Interval'], df_bar['Avg_Confidence']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{perc:.2f}%\n{count} clients\nAvg Churn: {avg_churn:.2f}\nChurn Interval: {interval}\nAvg Confidence: {avg_conf:.2f}%', ha='center', va='bottom')

    plt.title('Client Churn Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of Clients')
    plt.ylim(0, 100)  # Set y-axis range to 0-100%

    plt.tight_layout()  # Adjust layout to ensure everything fits well
    plt.savefig(output_file)
    plt.close()

# Main function
def main(file_path, plot_output_file, table_output_file, bar_chart_output_file):
    data = load_and_preprocess_data(file_path)
    churn_data = compute_churn_scores(data)
    churn_data, kmeans = apply_kmeans_clustering(churn_data)
    create_and_save_pie_chart(churn_data, kmeans, plot_output_file)
    print_and_save_table(churn_data, data, table_output_file)
    create_and_save_bar_chart(churn_data, bar_chart_output_file)
    intervals = classify_clients(churn_data)
    with open(table_output_file, 'a') as f:
        f.write("\nChurn Index Intervals for each cluster:\n")
        for cluster, interval in intervals.items():
            f.write(f"{cluster}: {interval[0]:.2f} - {interval[1]:.2f}\n")

# File paths
file_path = '/Users/igormol/Desktop/analytics/clients/Marajo/sample/Marajo.csv'
plot_output_file = 'client_churn_clusters.png'
table_output_file = 'client_churn_table.txt'
bar_chart_output_file = 'client_churn_bar_chart.png'

# Run the main function
main(file_path, plot_output_file, table_output_file, bar_chart_output_file)
