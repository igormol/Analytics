# webinterface.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import streamlit as st
from sklearn.metrics import mean_squared_error
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
    # Compute RFM scores for each client in the dataset
    rfm_table = rfm_clients_compute_rfm(data)

    # Normalize the RFM scores to ensure consistency in scale
    rfm_table = rfm_clients_normalize_rfm(rfm_table)

    # Perform clustering to group clients based on their RFM scores
    rfm_table = rfm_clients_perform_clustering(rfm_table)

    # Merge the RFM analysis results with client information
    result_table = rfm_clients_merge_client_info(data, rfm_table)

    # Sort the result table by RFM_Score in descending order and remove duplicates based on Client_ID
    result_table = result_table.sort_values(by='RFM_Score', ascending=False).drop_duplicates(subset='Client_ID')

    # Save the RFM analysis results to a CSV file
    result_table.to_csv('Client_RFM_table.csv', index=False)

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

    st.write("## RFM Analysis")

    tab1, tab2 = st.tabs(["Tree Chart", "RFM Table"])

    with tab1:
        st.write("### Tree Chart")
        fig_tree = rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)

        if fig_tree:
            # Plot the Tree Chart in the Streamlit app
            st.pyplot(fig_tree)

            # Save the Tree Chart as a PNG file
            fig_tree.savefig("Client_RFM_tree_chart.png")

            # Provide Download Option for Tree Chart
            with open("Client_RFM_tree_chart.png", "rb") as file:
                st.download_button(
                    label="Download Tree Chart as PNG",
                    data=file,
                    file_name="Client_RFM_tree_chart.png",
                    mime="image/png"
                )
        else:
            st.error("Error: Tree chart figure could not be generated.")

    with tab2:
        st.write("### Client RFM Table")
        st.dataframe(result_table)

        # Provide Download Option for Result Table
        csv = result_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name='Client_RFM_table.csv',
            mime='text/csv'
        )

def run_abc_analysis(data):
    # Aggregate sales data for each product using the abc_analysis_aggregate_sales function
    product_sales = abc_analysis_aggregate_sales(data)

    # Calculate cumulative sales for each product using the abc_analysis_calculate_cumulative_sales function
    product_sales = abc_analysis_calculate_cumulative_sales(product_sales)

    # Create two tabs in the Streamlit interface named "Charts" and "Table"
    subtab1, subtab2 = st.tabs(["Charts", "Table"])

    # In the "Charts" tab
    with subtab1:
        # Plot a bar chart of the product sales using the abc_analysis_plot_bar_chart function
        bar_chart = abc_analysis_plot_bar_chart(product_sales)

        # Add a download button for the bar chart, allowing it to be downloaded as a PNG file
        st.download_button(
            label="Download bar chart as PNG",
            data=bar_chart,
            file_name='abc_bar_chart.png',
            mime='image/png'
        )

        # Plot a pie chart of the product sales using the abc_analysis_plot_pie_chart function
        pie_chart = abc_analysis_plot_pie_chart(product_sales)

        # Add a download button for the pie chart, allowing it to be downloaded as a PNG file
        st.download_button(
            label="Download pie chart as PNG",
            data=pie_chart,
            file_name='abc_pie_chart.png',
            mime='image/png'
        )

    # In the "Table" tab
    with subtab2:
        # Display the product sales data in a table, showing columns 'Product_Name', 'Class', and 'Metric'
        st.dataframe(product_sales[['Product_Name', 'Class', 'Metric']])

        # Convert the product sales data to CSV format
        csv = product_sales.to_csv(index=False).encode('utf-8')

        # Add a download button for the table, allowing it to be downloaded as a CSV file
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='abc_analysis_results.csv',
            mime='text/csv'
        )

def run_churn_analysis(data):
    # Set the title of the Streamlit app to "Client Churn Analysis"
    st.title("Client Churn Analysis")

    # Compute churn scores for the data using the churn_analysis_compute_churn_scores function
    churn_data = churn_analysis_compute_churn_scores(data)

    # Apply k-means clustering to the churn data using the churn_analysis_apply_kmeans_clustering function
    churn_data, kmeans = churn_analysis_apply_kmeans_clustering(churn_data)

    # Create three tabs in the Streamlit interface named "Pie Chart", "Bar Chart", and "Table of Results"
    tab1, tab2, tab3 = st.tabs(["Pie Chart", "Bar Chart", "Table of Results"])

    # In the "Pie Chart" tab
    with tab1:
        # Display a pie chart of the churn data using the churn_analysis_show_pie_chart function
        churn_analysis_show_pie_chart(churn_data, kmeans)

    # In the "Bar Chart" tab
    with tab2:
        # Display a bar chart of the churn data using the churn_analysis_show_bar_chart function
        churn_analysis_show_bar_chart(churn_data)

    # In the "Table of Results" tab
    with tab3:
        # Display a table of the churn data using the churn_analysis_show_table function
        churn_analysis_show_table(churn_data)

def run_recsys_clients_to_products(data):
    # Encode product IDs and obtain encoded product names using recsys_clients_to_products_encode_product_ids
    data, product_encoder = recsys_clients_to_products_encode_product_ids(data)

    # Get unique product names
    product_names = data['Product_Name'].unique()

    # Display a dropdown menu for selecting a product for client recommendations
    selected_product = st.selectbox("Select a product for client recommendations", ["Select a product"] + list(product_names))

    # If a product is selected
    if selected_product != "Select a product":
        # Aggregate duplicate records by client and product, summing the 'Monetary_Value'
        data = recsys_clients_to_products_aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')

        # Create an interaction matrix of clients and products with their corresponding sales values
        interaction_matrix = recsys_clients_to_products_create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
        interaction_matrix_values = interaction_matrix.values

        # Get the number of clients and products in the interaction matrix
        num_clients, num_products = interaction_matrix.shape
        all_client_ids = interaction_matrix.index.tolist()  # Use original Client_IDs from the data

        # Prepare data for training the matrix factorization model
        X = np.argwhere(interaction_matrix_values)
        y = interaction_matrix_values[X[:, 0], X[:, 1]]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build a matrix factorization model
        mf_model = recsys_clients_to_products_build_matrix_factorization_model(num_clients, num_products)

        # Train the matrix factorization model
        recsys_clients_to_products_train_matrix_factorization_model(mf_model, X_train, y_train)

        # Generate predictions for the test set
        y_pred = mf_model.predict([X_test[:, 0], X_test[:, 1]]).flatten()

        # Compute the mean MSE for the test set
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean MSE for the test set: {mse}")

        # Generate recommendations for all clients for the selected product
        all_predictions = recsys_clients_to_products_generate_client_recommendations(product_encoder, selected_product, mf_model, all_client_ids)

        # Filter out clients who have already bought the product
        product_encoded_id = product_encoder.transform([selected_product])[0]
        bought_clients = data[data['Product_ID'] == product_encoded_id]['Client_ID'].unique()
        all_predictions = all_predictions[~all_predictions['Client_ID'].isin(bought_clients)]

        # Sort the predictions by predicted rating in descending order and select the top 15 recommendations
        all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)
        top_recommendations = all_predictions.nlargest(15, 'Predicted_Rating')

        # Display the top 15 client recommendations for the selected product
        st.write("Top 15 Client Recommendations for ", selected_product)
        st.table(top_recommendations[['Client_ID', 'Predicted_Rating']])

        # Add an option to export recommendations as a CSV file
        csv = top_recommendations.to_csv(index=False)
        st.download_button(
            label="Download recommendations as CSV",
            data=csv,
            file_name='client_recommendations.csv',
            mime='text/csv',
        )

def run_recsys_products_to_clients(data):
    # Encode client and product IDs and obtain encoded client names and product names
    data, client_encoder, product_encoder = recsys_products_to_clients_encode_ids(data)

    # Get unique client names
    client_names = data['Client_ID'].unique()
    client_names = client_encoder.inverse_transform(client_names)

    # Display a dropdown menu for selecting a client for product recommendations
    selected_client = st.selectbox("Select a client for recommendations", ["Select a client"] + list(client_names))

    # If a client is selected
    if selected_client != "Select a client":
        # Aggregate duplicate records by client and product, summing the 'Monetary_Value'
        data = recsys_products_to_clients_aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')

        # Create an interaction matrix of clients and products with their corresponding sales values
        interaction_matrix = recsys_products_to_clients_create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
        interaction_matrix_values = interaction_matrix.values

        # Get the number of clients and products in the interaction matrix
        num_clients, num_products = interaction_matrix.shape

        # Prepare data for training the matrix factorization model
        X = np.argwhere(interaction_matrix_values)
        y = interaction_matrix_values[X[:, 0], X[:, 1]]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build a matrix factorization model
        mf_model = recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products)

        # Train the matrix factorization model
        recsys_products_to_clients_train_matrix_factorization_model(mf_model, X_train, y_train)

        # Predict ratings for the test set and compute the MSE for these predictions
        y_pred_test = mf_model.predict([X_test[:, 0], X_test[:, 1]]).flatten()
        mse_test = mean_squared_error(y_test, y_pred_test)
        st.write(f"Mean Squared Error (MSE) on the test set: {mse_test}")

        # Generate recommendations for the selected client
        all_predictions = recsys_products_to_clients_generate_recommendations(client_encoder, selected_client, mf_model, num_products)

        # Filter out products the client has already bought
        bought_products = data[data['Client_ID'] == client_encoder.transform([selected_client])[0]]['Product_ID'].unique()
        all_predictions = all_predictions[~all_predictions['Product_ID'].isin(bought_products)]

        # Merge product names with predicted ratings
        all_predictions = all_predictions.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID', how='left')

        # Sort the predictions by predicted rating in descending order and select the top 15 recommendations
        all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)
        top_recommendations = all_predictions.nlargest(15, 'Predicted_Rating')

        # Display the top 15 recommendations for the selected client
        st.write("Top 15 Recommendations for ", selected_client)
        st.table(top_recommendations[['Product_Name', 'Predicted_Rating']])

        # Add a button to export recommendations as a CSV file
        if st.button("Export Recommendations as CSV"):
            csv = top_recommendations.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name='recommendations.csv', mime='text/csv')

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
