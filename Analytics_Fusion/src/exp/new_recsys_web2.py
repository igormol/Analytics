import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import streamlit as st

def recsys_clients_to_products_load_data(path):
    path.seek(0)
    data = pd.read_csv(path)
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    data.dropna(subset=['Sales_Date'], inplace=True)
    data['Recency'] = (datetime.now() - data['Sales_Date']).dt.days
    return data

def recsys_clients_to_products_encode_product_ids(data):
    product_encoder = LabelEncoder()
    data['Product_ID'] = product_encoder.fit_transform(data['Product_Name'])
    return data, product_encoder

def recsys_clients_to_products_aggregate_duplicates(data, client_col, product_col, rating_col):
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    aggregated_data = aggregated_data.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on=product_col, how='left')
    return aggregated_data

def recsys_clients_to_products_create_interaction_matrix(data, client_col, product_col, rating_col):
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix

def recsys_clients_to_products_build_matrix_factorization_model(num_clients, num_products, embedding_dim=25, l2_reg=1e-6, dropout_rate=0.5):
    client_input = keras.layers.Input(shape=(1,), name='client_input')
    client_embedding = keras.layers.Embedding(input_dim=num_clients, output_dim=embedding_dim,
                                              embeddings_regularizer=keras.regularizers.l2(l2_reg),
                                              name='client_embedding')(client_input)
    client_vec = keras.layers.Flatten()(client_embedding)

    product_input = keras.layers.Input(shape=(1,), name='product_input')
    product_embedding = keras.layers.Embedding(input_dim=num_products, output_dim=embedding_dim,
                                               embeddings_regularizer=keras.regularizers.l2(l2_reg),
                                               name='product_embedding')(product_input)
    product_vec = keras.layers.Flatten()(product_embedding)

    concatenated = keras.layers.Concatenate()([client_vec, product_vec])
    dropout = keras.layers.Dropout(dropout_rate)(concatenated)
    output = keras.layers.Dense(1)(dropout)

    model = keras.models.Model(inputs=[client_input, product_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def recsys_clients_to_products_train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def recsys_clients_to_products_generate_client_recommendations(product_encoder, specific_product_name, model, all_client_ids):
    specific_product_encoded_id = product_encoder.transform([specific_product_name])[0]
    product_client_combinations = np.array([(i, specific_product_encoded_id) for i in range(len(all_client_ids))])

    client_ids_encoded = product_client_combinations[:, 0]
    product_ids = product_client_combinations[:, 1]
    predictions = model.predict([client_ids_encoded, product_ids]).flatten()

    scaler = MinMaxScaler(feature_range=(0, 100))
    predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

    return pd.DataFrame({
        'Product_Name': [specific_product_name] * len(predictions_normalized),
        'Client_ID_Encoded': client_ids_encoded,
        'Client_ID': all_client_ids,
        'Predicted_Rating': predictions_normalized
    })

def run_recsys_clients_to_products(data):
    data, product_encoder = recsys_clients_to_products_encode_product_ids(data)
    product_names = data['Product_Name'].unique()

    selected_product = st.selectbox("Select a product for client recommendations", ["Select a product"] + list(product_names))

    if selected_product != "Select a product":
        data = recsys_clients_to_products_aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
        interaction_matrix = recsys_clients_to_products_create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
        interaction_matrix_values = interaction_matrix.values

        num_clients, num_products = interaction_matrix.shape
        all_client_ids = interaction_matrix.index.tolist()  # Use original Client_IDs from the data

        X = np.argwhere(interaction_matrix_values)
        y = interaction_matrix_values[X[:, 0], X[:, 1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mf_model = recsys_clients_to_products_build_matrix_factorization_model(num_clients, num_products)
        recsys_clients_to_products_train_matrix_factorization_model(mf_model, X_train, y_train)

        all_predictions = recsys_clients_to_products_generate_client_recommendations(product_encoder, selected_product, mf_model, all_client_ids)

        # Filter out clients who have already bought the product
        product_encoded_id = product_encoder.transform([selected_product])[0]
        bought_clients = data[data['Product_ID'] == product_encoded_id]['Client_ID'].unique()
        all_predictions = all_predictions[~all_predictions['Client_ID'].isin(bought_clients)]

        all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)

        top_recommendations = all_predictions.nlargest(15, 'Predicted_Rating')

        st.write("Top 15 Client Recommendations for ", selected_product)
        st.table(top_recommendations[['Client_ID', 'Predicted_Rating']])

        # Add option to export recommendations as CSV
        csv = top_recommendations.to_csv(index=False)
        st.download_button(
            label="Download recommendations as CSV",
            data=csv,
            file_name='client_recommendations.csv',
            mime='text/csv',
        )

def main():
    st.title("Client Recommendation System")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = recsys_clients_to_products_load_data(uploaded_file)

        tab = st.sidebar.selectbox("Select Tab", ["Recommender System"])

        if tab == "Recommender System":
            run_recsys_clients_to_products(data)

if __name__ == "__main__":
    main()
