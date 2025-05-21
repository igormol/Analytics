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

def recsys_products_to_clients_generate_recommendations(client_encoder, product_encoder, model, num_products, data):
    all_recommendations = []

    for client_encoded_id in range(len(client_encoder.classes_)):
        client_product_combinations = np.array([(client_encoded_id, product) for product in range(num_products)])

        all_client_ids = client_product_combinations[:, 0]
        all_product_ids = client_product_combinations[:, 1]
        predictions = model.predict([all_client_ids, all_product_ids]).flatten()

        scaler = MinMaxScaler(feature_range=(0, 100))
        predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

        client_recommendations = pd.DataFrame({
            'Client_ID': client_encoded_id,
            'Product_ID': all_product_ids,
            'Predicted_Rating': predictions_normalized
        })

        # Filter out products the client has already bought
        bought_products = data[data['Client_ID'] == client_encoded_id]['Product_ID'].unique()
        client_recommendations = client_recommendations[~client_recommendations['Product_ID'].isin(bought_products)]

        # Get top 10 recommendations
        top_recommendations = client_recommendations.sort_values(by='Predicted_Rating', ascending=False).head(10)
        all_recommendations.append(top_recommendations)

    all_recommendations_df = pd.concat(all_recommendations, ignore_index=True)

    # Ensure product names are merged correctly
    all_recommendations_df['Client_ID'] = client_encoder.inverse_transform(all_recommendations_df['Client_ID'])
    all_recommendations_df['Product_ID'] = product_encoder.inverse_transform(all_recommendations_df['Product_ID'])

    # Merge to get product names
    all_recommendations_df = all_recommendations_df.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID', how='left')

    return all_recommendations_df

def run_recsys_products_to_clients(data):
    data, client_encoder, product_encoder = recsys_products_to_clients_encode_ids(data)

    data = recsys_products_to_clients_aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
    interaction_matrix = recsys_products_to_clients_create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
    interaction_matrix_values = interaction_matrix.values

    num_clients, num_products = interaction_matrix.shape

    X = np.argwhere(interaction_matrix_values)
    y = interaction_matrix_values[X[:, 0], X[:, 1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mf_model = recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products)
    recsys_products_to_clients_train_matrix_factorization_model(mf_model, X_train, y_train)

    # Generate recommendations for all clients
    all_recommendations = recsys_products_to_clients_generate_recommendations(client_encoder, product_encoder, mf_model, num_products, data)

    st.write("Recommendations for All Clients")
    st.dataframe(all_recommendations[['Client_ID', 'Product_Name', 'Predicted_Rating']])

    # Export recommendations to CSV
    if st.button("Export All Recommendations as CSV"):
        csv = all_recommendations.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name='all_recommendations.csv', mime='text/csv')

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
