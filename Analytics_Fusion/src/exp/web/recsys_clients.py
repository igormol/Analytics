import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import defaultdict

# Load and preprocess data
def recsys_clients_load_and_preprocess_data(file):
    data = pd.read_csv(file)
    client_name_map = data[['Client_ID']].drop_duplicates().reset_index(drop=True)
    client_name_map['Client_Name'] = client_name_map['Client_ID']

    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])

    return data, client_name_map, client_encoder, product_encoder

# Split data
def recsys_clients_split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

# Convert date to numerical format
def recsys_clients_convert_date_to_numerical(data):
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date']).astype(int) / 10**9
    return data

# Train model
def recsys_clients_train_model(train, features, target):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(train[features], train[target])
    return model

# Predict and evaluate
def recsys_clients_predict_and_evaluate(model, test, features, target):
    test['Predicted_Value'] = model.predict(test[features])
    mse = mean_squared_error(test[target], test['Predicted_Value'])
    return test, mse

# Normalize scores
def recsys_clients_normalize_scores(df):
    min_score = df['Predicted_Value'].min()
    max_score = df['Predicted_Value'].max()
    df['Normalized_Score'] = 100 * (df['Predicted_Value'] - min_score) / (max_score - min_score)
    return df

# Generate recommendations
def recsys_clients_generate_recommendations_reverse(test, top_k):
    recommendations = defaultdict(list)
    products = test['Product_ID'].unique()
    for product in products:
        product_data = test[test['Product_ID'] == product]
        product_recommendations = product_data.sort_values(by='Normalized_Score', ascending=False)
        unique_recommendations = product_recommendations.drop_duplicates(subset='Client_ID').head(top_k)
        recommendations[product] = list(unique_recommendations['Client_ID'])
    return recommendations

# Map product IDs to names
def recsys_clients_map_product_ids_to_names(data):
    return dict(zip(data['Product_ID'], data['Product_Name']))

# Decode client IDs to names
def recsys_clients_decode_client_ids_to_names(client_name_map, client_encoder):
    client_name_map['Client_ID'] = client_encoder.transform(client_name_map['Client_Name'])
    return client_name_map

# Calculate recall at k
def recsys_clients_recall_at_k(recommendations, actuals, k):
    recalls = {}
    for product in recommendations:
        if product in actuals:  # Added check to ensure the product exists in actuals
            recommended_clients = recommendations[product][:k]
            actual_clients = actuals[product]
            recall = len(set(recommended_clients) & set(actual_clients)) / len(actual_clients) if len(actual_clients) > 0 else 0
            recalls[product] = recall
    return recalls

# Functionality for "Recommendations of Clients to Products"
def recsys_clients_recommendations(recsys_clients_data, recsys_clients_client_name_map, recsys_clients_client_encoder, recsys_clients_product_encoder):
    train, test = recsys_clients_split_data(recsys_clients_data)

    train = recsys_clients_convert_date_to_numerical(train)
    test = recsys_clients_convert_date_to_numerical(test)

    features = ['Sales_Date', 'Client_ID', 'Product_ID']
    target = 'Monetary_Value'

    model = recsys_clients_train_model(train, features, target)
    test, mse = recsys_clients_predict_and_evaluate(model, test, features, target)
    st.write(f'Mean Squared Error: {mse}')

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

# Main function
def main():
    st.title("Product Recommendation System")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        recsys_clients_data, recsys_clients_client_name_map, recsys_clients_client_encoder, recsys_clients_product_encoder = recsys_clients_load_and_preprocess_data(uploaded_file)

        # Sidebar navigation
        st.sidebar.title("Navigation")
        tab = st.sidebar.radio("Go to", ["Recommendations of Clients to Products", "Another Functionality"])

        if tab == "Recommendations of Clients to Products":
            recsys_clients_recommendations(recsys_clients_data, recsys_clients_client_name_map, recsys_clients_client_encoder, recsys_clients_product_encoder)
        elif tab == "Another Functionality":
            st.write("This is where another functionality will be implemented.")
            # Call another function here when implemented

if __name__ == "__main__":
    main()
