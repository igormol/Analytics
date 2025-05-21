import streamlit as st
from rfm_products import *
from rfm_clients import *
from abc_analysis import *
from churn_analysis import *
from recsys_clients import *
from recsys_products import *

def run_rfm_products(data):
    rfm_table = rfm_products_compute_rfm(data)
    rfm_table = rfm_products_normalize_rfm(rfm_table)
    rfm_table = rfm_products_perform_clustering(rfm_table)
    result_table = rfm_products_merge_product_info(data, rfm_table)

    # Sort by RFM_Score in descending order and remove duplicate products
    result_table = result_table.sort_values(by='RFM_Score', ascending=False).drop_duplicates(subset='Product_ID')

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

    tab1, tab2, tab3 = st.tabs(["Tree Chart", "Pie Chart", "RFM Table"])

    with tab1:
        cluster_counts = result_table['Cluster'].value_counts()
        mean_rfm_scores = result_table.groupby('Cluster')['RFM_Score'].mean()
        confidences = result_table.groupby('Cluster')['Confidence_Interval'].mean()
        fig_tree = rfm_products_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)
        st.pyplot(fig_tree)
        tree_chart_file = "tree_chart.png"
        fig_tree.savefig(tree_chart_file)
        with open(tree_chart_file, "rb") as file:
            st.download_button("Download Tree Chart", file, "tree_chart.png")

    with tab2:
        cluster_counts = result_table['Cluster'].value_counts()
        fig_pie = rfm_products_plot_pie_chart(cluster_counts, colors)
        st.pyplot(fig_pie)
        pie_chart_file = "pie_chart.png"
        fig_pie.savefig(pie_chart_file)
        with open(pie_chart_file, "rb") as file:
            st.download_button("Download Pie Chart", file, "pie_chart.png")

    with tab3:
        rfm_products_display_table(result_table)
        csv_file = "Product_RFM_table.csv"
        result_table.to_csv(csv_file, index=False)
        with open(csv_file, "rb") as file:
            st.download_button("Download RFM Table", file, "Product_RFM_table.csv")

def run_rfm_clients(data):
    rfm_table = rfm_clients_compute_rfm(data)
    rfm_table = rfm_clients_normalize_rfm(rfm_table)
    rfm_table = rfm_clients_perform_clustering(rfm_table)
    result_table = rfm_clients_merge_client_info(data, rfm_table)

    # Sort by RFM_Score in descending order and remove duplicate clients
    result_table = result_table.sort_values(by='RFM_Score', ascending=False).drop_duplicates(subset='Client_ID')

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
        rfm_clients_plot_tree_chart(cluster_counts, mean_rfm_scores, colors, confidences)

        # Provide Download Option for Tree Chart
        with open("Client_RFM_tree_chart.png", "rb") as file:
            st.download_button(
                label="Download Tree Chart as PNG",
                data=file,
                file_name="Client_RFM_tree_chart.png",
                mime="image/png"
            )

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

    # Save Table to File
    rfm_clients_save_table_to_file(result_table)

def run_abc_analysis(data):
    product_sales = abc_analysis_aggregate_sales(data)
    product_sales = abc_analysis_calculate_cumulative_sales(product_sales)

    subtab1, subtab2 = st.tabs(["Charts", "Table"])

    with subtab1:
        bar_chart = abc_analysis_plot_bar_chart(product_sales)

        st.download_button(
            label="Download bar chart as PNG",
            data=bar_chart,
            file_name='abc_bar_chart.png',
            mime='image/png'
        )

        pie_chart = abc_analysis_plot_pie_chart(product_sales)

        st.download_button(
            label="Download pie chart as PNG",
            data=pie_chart,
            file_name='abc_pie_chart.png',
            mime='image/png'
        )

    with subtab2:
        st.dataframe(product_sales[['Product_Name', 'Class', 'Metric']])

        # Add a download button for the table as CSV
        csv = product_sales.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='abc_analysis_results.csv',
            mime='text/csv'
        )

def run_churn_analysis(data):
    st.title("Client Churn Analysis")
    churn_data = churn_analysis_compute_churn_scores(data)
    churn_data, kmeans = churn_analysis_apply_kmeans_clustering(churn_data)

    tab1, tab2, tab3 = st.tabs(["Pie Chart", "Bar Chart", "Table of Results"])

    with tab1:
        churn_analysis_show_pie_chart(churn_data, kmeans)

    with tab2:
        churn_analysis_show_bar_chart(churn_data)

    with tab3:
        churn_analysis_show_table(churn_data)

# Functionality for "Recommendations of Clients to Products"
def recsys_clients_recommendations(recsys_clients_data, recsys_clients_client_name_map, recsys_clients_client_encoder, recsys_clients_product_encoder):
    train, test = recsys_clients_split_data(recsys_clients_data)

    train = recsys_clients_convert_date_to_numerical(train)
    test = recsys_clients_convert_date_to_numerical(test)

    features = ['Sales_Date', 'Client_ID', 'Product_ID']
    target = 'Monetary_Value'

    model = recsys_clients_train_model(train, features, target)
    test, mse = recsys_clients_predict_and_evaluate(model, test, features, target)

    test = recsys_clients_normalize_scores(test)

    top_k = 10
    recommendations = recsys_clients_generate_recommendations_reverse(test, top_k)

    product_map = recsys_clients_map_product_ids_to_names(recsys_clients_data)
    recsys_clients_client_name_map = recsys_clients_decode_client_ids_to_names(recsys_clients_client_name_map, recsys_clients_client_encoder)

    products = test['Product_ID'].unique()
    product_names = [product_map[product] for product in products]

    selected_product_name = st.selectbox("Select a product to see recommendations", options=["All"] + product_names)

    if selected_product_name != "All":
        selected_product_id = [product for product, name in product_map.items() if name == selected_product_name][0]
        filtered_recommendations = {selected_product_id: recommendations[selected_product_id]}
        actual_clients = test[test['Product_ID'] == selected_product_id]['Client_ID'].unique()
        recall = recsys_clients_recall_at_k(filtered_recommendations, {selected_product_id: actual_clients}, top_k)
    else:
        filtered_recommendations = recommendations
        actuals = {product: test[test['Product_ID'] == product]['Client_ID'].unique() for product in products}
        recall = recsys_clients_recall_at_k(recommendations, actuals, top_k)

    for product, clients in filtered_recommendations.items():
        product_name = product_map[product]
        st.write(f"Product: {product_name}")
        rec_table = []
        for client in clients:
            client_name = recsys_clients_client_name_map[recsys_clients_client_name_map['Client_ID'] == client]['Client_Name'].values[0]
            predicted_score = test[(test['Product_ID'] == product) & (test['Client_ID'] == client)]['Normalized_Score'].values[0]
            rec_table.append([client_name, predicted_score])
        rec_table = sorted(rec_table, key=lambda x: x[1], reverse=True)
        st.write(f"Recall at {top_k}: {recall[product]:.2f}" if product in recall else f"Recall at {top_k}: N/A")
        st.table(pd.DataFrame(rec_table, columns=["Client Name", "Predicted Score"]))

def recsys_products_main_functionality(data, client_name_map, client_encoder, product_encoder):
    train, test = recsys_products_split_data(data)

    train = recsys_products_convert_date_to_numerical(train)
    test = recsys_products_convert_date_to_numerical(test)

    features = ['Sales_Date', 'Client_ID', 'Product_ID']
    target = 'Monetary_Value'

    model = recsys_products_train_model(train, features, target)
    test, mse = recsys_products_predict_and_evaluate(model, test, features, target)
    st.write(f'Mean Squared Error: {mse}')

    test = recsys_products_normalize_scores(test)

    top_k = 10
    recommendations = recsys_products_generate_recommendations(test, top_k)

    average_recall = recsys_products_calculate_recall_scores(test, recommendations, top_k)
    st.write(f'Average Recall at {top_k}: {average_recall}')

    product_map = recsys_products_map_product_ids_to_names(data)
    client_name_map = recsys_products_decode_client_ids_to_names(client_name_map, client_encoder)

    clients = test['Client_ID'].unique()
    client_names = client_name_map['Client_Name'].unique()

    client_name = st.selectbox('Select a Client', ['All Clients'] + list(client_names))

    if client_name == 'All Clients':
        for client in clients:
            client_name = client_name_map[client_name_map['Client_ID'] == client]['Client_Name'].values[0]
            st.subheader(f'Recommendations for {client_name}')
            client_data = test[test['Client_ID'] == client]
            client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False).drop_duplicates(subset='Product_ID').head(top_k)
            recommendations_list = [{"Product Name": product_map[row['Product_ID']], "Predicted Score": row['Normalized_Score']} for _, row in client_recommendations.iterrows()]
            st.table(pd.DataFrame(recommendations_list))
    else:
        client_id = client_name_map[client_name_map['Client_Name'] == client_name]['Client_ID'].values[0]
        st.subheader(f'Recommendations for {client_name}')
        client_data = test[test['Client_ID'] == client_id]
        client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False).drop_duplicates(subset='Product_ID').head(top_k)
        recommendations_list = [{"Product Name": product_map[row['Product_ID']], "Predicted Score": row['Normalized_Score']} for _, row in client_recommendations.iterrows()]
        st.table(pd.DataFrame(recommendations_list))
