import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import streamlit as st

def recsys_products_to_clients_load_data(path):
    data = pd.read_csv(path)
    data['Brand_Name'].fillna('Unknown', inplace=True)
    return data

def recsys_products_to_clients_encode_ids(data):
    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])
    return data, client_encoder, product_encoder

def recsys_products_to_clients_aggregate_duplicates(data, client_col, product_col, rating_col):
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    aggregated_data = aggregated_data.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on=product_col, how='left')
    return aggregated_data

def recsys_products_to_clients_create_interaction_matrix(data, client_col, product_col, rating_col):
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix

def recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products, embedding_dim=25, l2_reg=1e-6, dropout_rate=0.5):
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

def recsys_products_to_clients_train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def recsys_products_to_clients_generate_recommendations(client_encoder, specific_client_id, model, num_products):
    specific_client_encoded_id = client_encoder.transform([specific_client_id])[0]
    client_product_combinations = np.array([(specific_client_encoded_id, product) for product in range(num_products)])

    all_client_ids = client_product_combinations[:, 0]
    all_product_ids = client_product_combinations[:, 1]
    predictions = model.predict([all_client_ids, all_product_ids]).flatten()

    scaler = MinMaxScaler(feature_range=(0, 100))
    predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

    return pd.DataFrame({
        'Client_ID': [specific_client_id] * len(predictions_normalized),
        'Product_ID': all_product_ids,
        'Predicted_Rating': predictions_normalized
    })

def run_recsys_products_to_clients(data):
    data, client_encoder, product_encoder = recsys_products_to_clients_encode_ids(data)
    client_names = data['Client_ID'].unique()
    client_names = client_encoder.inverse_transform(client_names)

    selected_client = st.selectbox("Select a client for recommendations", ["Select a client"] + list(client_names))

    if selected_client != "Select a client":
        data = recsys_products_to_clients_aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
        interaction_matrix = recsys_products_to_clients_create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
        interaction_matrix_values = interaction_matrix.values

        num_clients, num_products = interaction_matrix.shape

        X = np.argwhere(interaction_matrix_values)
        y = interaction_matrix_values[X[:, 0], X[:, 1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mf_model = recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products)
        recsys_products_to_clients_train_matrix_factorization_model(mf_model, X_train, y_train)

        all_predictions = recsys_products_to_clients_generate_recommendations(client_encoder, selected_client, mf_model, num_products)

        # Filter out products the client has already bought
        bought_products = data[data['Client_ID'] == client_encoder.transform([selected_client])[0]]['Product_ID'].unique()
        all_predictions = all_predictions[~all_predictions['Product_ID'].isin(bought_products)]

        all_predictions = all_predictions.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID', how='left')
        all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)

        top_recommendations = all_predictions.nlargest(15, 'Predicted_Rating')

        st.write("Top 15 Recommendations for ", selected_client)
        st.table(top_recommendations[['Product_Name', 'Predicted_Rating']])

        # Export recommendations to CSV
        if st.button("Export Recommendations as CSV"):
            csv = top_recommendations.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name='recommendations.csv', mime='text/csv')

def main():
    st.title("Product Recommendation System")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = recsys_products_to_clients_load_data(uploaded_file)

        tab = st.sidebar.selectbox("Select Tab", ["Recommender System"])

        if tab == "Recommender System":
            run_recsys_products_to_clients(data)

if __name__ == "__main__":
    main()
