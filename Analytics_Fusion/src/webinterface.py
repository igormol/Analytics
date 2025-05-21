# webinterface.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import streamlit as st
from rfm_products import *
from rfm_clients import *
from abc_analysis import *
from churn_analysis import *
from recsys_products_to_clients import *
from recsys_clients_to_products import *
from recsys_products_to_products import *

def run_rfm_products(data):
    # Compute RFM scores for each product in the dataset
    rfm_table = rfm_products_compute_rfm(data)

    # Normalize the RFM scores to ensure consistency in scale
    rfm_table = rfm_products_normalize_rfm(rfm_table)

    # Perform clustering to group products based on their RFM scores
    rfm_table = rfm_products_perform_clustering(rfm_table)

    # Merge the RFM analysis results with product information
    result_table = rfm_products_merge_product_info(data, rfm_table)

    # Sort the result table by RFM_Score in descending order and remove duplicates based on Product_ID
    result_table = result_table.sort_values(by='RFM_Score', ascending=False).drop_duplicates(subset='Product_ID')

    # Define a color mapping for different clusters
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

    # Display a header for the RFM Analysis
    st.write("## RFM Analysis")

    # Create tabs for different visualizations: Tree Chart, Pie Chart, RFM Table
    tab1, tab2, tab3 = st.tabs(["Tree Chart", "Pie Chart", "RFM Table"])

    # Tree Chart visualization
    with tab1:
        # Calculate cluster counts, mean RFM scores, and confidence intervals
        cluster_counts = result_table['Cluster'].value_counts()
        mean_rfm_scores = result_table.groupby('Cluster')['RFM_Score'].mean()
        confidences = result_table.groupby('Cluster')['Confidence_Interval'].mean()

        # Plot the Tree Chart
        fig_tree = rfm_products_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)
        st.pyplot(fig_tree)

        # Save the Tree Chart as an image and provide a download button
        tree_chart_file = "tree_chart.png"
        fig_tree.savefig(tree_chart_file)
        with open(tree_chart_file, "rb") as file:
            st.download_button("Download Tree Chart", file, "tree_chart.png")

    # Pie Chart visualization
    with tab2:
        # Calculate cluster counts
        cluster_counts = result_table['Cluster'].value_counts()

        # Plot the Pie Chart
        fig_pie = rfm_products_plot_pie_chart(cluster_counts, colors)
        st.pyplot(fig_pie)

        # Save the Pie Chart as an image and provide a download button
        pie_chart_file = "pie_chart.png"
        fig_pie.savefig(pie_chart_file)
        with open(pie_chart_file, "rb") as file:
            st.download_button("Download Pie Chart", file, "pie_chart.png")

    # RFM Table visualization
    with tab3:
        # Display the RFM analysis results in tabular format
        rfm_products_display_table(result_table)

        # Save the RFM Table as a CSV file and provide a download button
        csv_file = "Product_RFM_table.csv"
        result_table.to_csv(csv_file, index=False)
        with open(csv_file, "rb") as file:
            st.download_button("Download RFM Table", file, "Product_RFM_table.csv")

def run_rfm_clients(data):
    # Compute RFM scores for each client in the dataset.
    rfm_table = rfm_clients_compute_rfm(data)

    # Normalize the RFM scores to ensure consistency in scale.
    rfm_table = rfm_clients_normalize_rfm(rfm_table)

    # Perform clustering to group clients based on their RFM scores.
    rfm_table = rfm_clients_perform_clustering(rfm_table)

    # Merge the RFM analysis results with client information.
    result_table = rfm_clients_merge_client_info(data, rfm_table)

    # Sort the result table by RFM_Score in descending order and remove duplicates based on Client_ID.
    result_table = result_table.sort_values(by='RFM_Score', ascending=False).drop_duplicates(subset='Client_ID')

    # Save the RFM analysis results to a CSV file.
    result_table.to_csv('Client_RFM_table.csv', index=False)

    # Calculate the counts of each cluster, mean RFM scores, and confidence intervals.
    cluster_counts = result_table['Cluster'].value_counts()
    mean_rfm_scores = result_table.groupby('Cluster')['RFM_Score'].mean()
    confidences = result_table.groupby('Cluster')['Confidence_Interval'].mean()

    # Define colors for different client clusters.
    colors = {
        'Loyal-client': '#FFD700',          # Gold
        'Gold-client': '#FFA500',           # Orange
        'VIP-client': '#FF4500',            # OrangeRed
        'Promising Customer': '#ADFF2F',    # GreenYellow
        'Cooling-down-client': '#00CED1',   # DarkTurquoise
        'Hibernating': '#4682B4',           # SteelBlue
        'Client-at-risk': '#FF6347',        # Tomato
        'New-client': '#32CD32'             # LimeGreen
    }

    # Display the RFM Analysis section title.
    st.write("## RFM Analysis")

    # Create two tabs in the Streamlit app for different visualizations.
    tab1, tab2 = st.tabs(["Tree Chart", "RFM Table"])

    # Display a tree chart of the RFM analysis in the first tab.
    with tab1:
        st.write("### Tree Chart")
        fig_tree = rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)

        if fig_tree:
            # Plot the tree chart in the Streamlit app.
            st.pyplot(fig_tree)

            # Save the tree chart as a PNG file.
            fig_tree.savefig("Client_RFM_tree_chart.png")

            # Provide a download option for the tree chart.
            with open("Client_RFM_tree_chart.png", "rb") as file:
                st.download_button(
                    label="Download Tree Chart as PNG",
                    data=file,
                    file_name="Client_RFM_tree_chart.png",
                    mime="image/png"
                )
        else:
            # Display an error message if the tree chart could not be generated.
            st.error("Error: Tree chart figure could not be generated.")

    # Display the client RFM table in the second tab.
    with tab2:
        st.write("### Client RFM Table")
        st.dataframe(result_table)

        # Provide a download option for the result table as a CSV file.
        csv = result_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name='Client_RFM_table.csv',
            mime='text/csv'
        )

def run_abc_analysis(data):
    # Aggregate sales data for each product.
    product_sales = abc_analysis_aggregate_sales(data)

    # Calculate cumulative sales and classify products into ABC categories.
    product_sales = abc_analysis_calculate_cumulative_sales(product_sales)

    # Create two tabs in the Streamlit app for different visualizations.
    subtab1, subtab2 = st.tabs(["Charts", "Table"])

    # Display bar and pie charts of the ABC analysis in the first tab.
    with subtab1:
        # Plot a bar chart of the product sales.
        bar_chart = abc_analysis_plot_bar_chart(product_sales)

        # Add a download button for the bar chart as a PNG file.
        st.download_button(
            label="Download bar chart as PNG",
            data=bar_chart,
            file_name='abc_bar_chart.png',
            mime='image/png'
        )

        # Plot a pie chart of the product sales.
        pie_chart = abc_analysis_plot_pie_chart(product_sales)

        # Add a download button for the pie chart as a PNG file.
        st.download_button(
            label="Download pie chart as PNG",
            data=pie_chart,
            file_name='abc_pie_chart.png',
            mime='image/png'
        )

    # Display the product sales data in a table format in the second tab.
    with subtab2:
        st.dataframe(product_sales[['Product_Name', 'Class', 'Metric']])

        # Add a download button for the table data as a CSV file.
        csv = product_sales.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='abc_analysis_results.csv',
            mime='text/csv'
        )

def run_churn_analysis(data):
    # Display the title for the Streamlit app section.
    st.title("Client Churn Analysis")

    # Compute churn scores for the data.
    churn_data = churn_analysis_compute_churn_scores(data)

    # Apply KMeans clustering to the churn data and return the clustered data and the KMeans model.
    churn_data, kmeans = churn_analysis_apply_kmeans_clustering(churn_data)

    # Create three tabs in the Streamlit app for different visualizations.
    tab1, tab2, tab3 = st.tabs(["Pie Chart", "Bar Chart", "Table of Results"])

    # Display a pie chart of the churn data in the first tab.
    with tab1:
        churn_analysis_show_pie_chart(churn_data, kmeans)

    # Display a bar chart of the churn data in the second tab.
    with tab2:
        churn_analysis_show_bar_chart(churn_data)

    # Display a table of the churn data in the third tab.
    with tab3:
        churn_analysis_show_table(churn_data)

def run_recsys_clients_to_products(data):
    # Display the title for the Streamlit app section.
    st.title("Hybrid Recommendation System")

    # Provide instructions for the user to upload a CSV file and select a product.
    st.write("Upload your CSV file and choose a product to get client recommendations.")

    # Preprocess the data to encode user and item IDs and handle missing values.
    data, user_encoder, item_encoder = recsys_clients_to_products_preprocess_data(data)

    # Extract unique product names from the preprocessed data.
    product_names = data['Product_Name'].unique()

    # Create a dropdown selection box for choosing a product, with an initial "Select a product" option.
    product_name = st.selectbox("Select a product", ["Select a product"] + list(product_names))

    # Check if a valid product is selected.
    if product_name and product_name != "Select a product":
        # Aggregate data to sum up 'Monetary_Value' by user and item IDs.
        data_aggregated = recsys_clients_to_products_aggregate_data(data)

        # Create an interaction matrix where rows are items and columns are users.
        interaction_matrix = recsys_clients_to_products_create_interaction_matrix(data_aggregated)

        # Split the interaction matrix into training and testing sets.
        train_data, test_data = recsys_clients_to_products_train_test_split(interaction_matrix)

        # Get the number of users (columns) in the interaction matrix.
        input_dim = interaction_matrix.shape[1]

        # Train an autoencoder on the training data and get the encoded training data.
        encoded_train_data, encoder = recsys_clients_to_products_train_autoencoder(train_data, test_data, input_dim)

        # Train an XGBoost model using the encoded training data.
        model = recsys_clients_to_products_train_xgboost(encoded_train_data, train_data)

        # Generate the top 10 client recommendations for the selected product.
        top_10_recommendations, mse_per_user, mean_mse = recsys_clients_to_products_generate_recommendations(product_name, item_encoder, encoded_train_data, model, user_encoder, interaction_matrix, data)

        # Display the top 10 client recommendations for the selected product.
        st.write("Top 10 Client Recommendations for ", product_name)
        st.table(top_10_recommendations[['Client_ID', 'predicted_rating']])

        # Display the MSE for each user and the mean MSE for all recommendations.
        st.write("Mean Squared Error for each user:")
        st.table(mse_per_user)
        st.write(f"Mean MSE for all recommendations: {mean_mse:.4f}")

def run_hybrid_recsys(data):
    st.title("Hybrid Recommendation System")
    st.write("Upload your CSV file and choose a client to get recommendations.")

    data, user_encoder, item_encoder = hybrid_recsys_preprocess_data(data)
    client_names = data['Client_ID'].unique()
    client_name = st.selectbox("Select a client", ["Select a client"] + list(client_names))

    if client_name and client_name != "Select a client":
        data_aggregated = hybrid_recsys_aggregate_data(data)
        interaction_matrix = hybrid_recsys_create_interaction_matrix(data_aggregated)
        train_data, test_data = hybrid_recsys_train_test_split(interaction_matrix)
        input_dim = interaction_matrix.shape[1]
        encoded_train_data, encoder = hybrid_recsys_train_autoencoder(train_data, test_data, input_dim)
        model = hybrid_recsys_train_xgboost(encoded_train_data, train_data)

        top_10_recommendations = hybrid_recsys_generate_recommendations(client_name, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix, data)
        st.write("Top 10 Recommendations for ", client_name)
        st.table(top_10_recommendations[['Product_ID', 'Product_Name', 'predicted_rating']])

        # Compute and display recall at K and MSE for the selected client
        recall_at_k = compute_recall_at_k(client_name, top_10_recommendations, test_data, user_encoder, item_encoder, k=10)
        mse = compute_mse(client_name, top_10_recommendations, test_data, user_encoder, item_encoder)
        st.write(f"Recall@10 for {client_name}: {recall_at_k:.4f}")
        st.write(f"MSE for {client_name}: {mse:.4f}")

        # Compute and display the average training MSE for all clients
        avg_train_mse = compute_avg_train_mse(client_names, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix)
        st.write(f"Average Training MSE for all clients: {avg_train_mse:.4f}")

def run_recsys_product_to_product(df):
    # Display a header for the Streamlit app section.
    st.header("Frequently Bought Items")

    # Extract unique product names from the DataFrame.
    product_names = df['Product_Name'].unique()

    # Create a dropdown selection box for choosing a product, with an empty initial option.
    selected_product = st.selectbox("Select a product to get recommendations", [""] + list(product_names))

    # Check if a valid product is selected.
    if selected_product and selected_product != "":
        # Create a user-item matrix with 'Monetary_Value' as values.
        user_item_matrix = recsys_products_to_products_create_user_item_matrix(df)

        # Transpose the user-item matrix to get a product-user matrix.
        product_user_matrix = user_item_matrix.T

        # Fit a K-Nearest Neighbors model using the product-user matrix.
        model_knn = recsys_products_to_products_fit_knn_model(product_user_matrix)

        # Get the top N product recommendations based on the selected product.
        recommendations = recsys_products_to_products_get_top_n_recommendations(selected_product, product_user_matrix, model_knn)

        # Check if there are recommendations available.
        if recommendations:
            # Display the recommendations in a table format.
            st.write(f"Recommendations for '{selected_product}':")
            st.table(pd.DataFrame(recommendations, columns=["Recommended Product", "Distance"]))
        else:
            # Display a message if no recommendations are found.
            st.write(f"No recommendations found for '{selected_product}'.")
