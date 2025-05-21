import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb

def load_data(path):
    data = pd.read_csv(path)
    data['Brand_Name'].fillna('Unknown', inplace=True)
    return data

def encode_ids(data):
    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])
    return data, client_encoder, product_encoder

def aggregate_duplicates(data, client_col, product_col, rating_col):
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    aggregated_data = aggregated_data.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on=product_col, how='left')
    return aggregated_data

def create_interaction_matrix(data, client_col, product_col, rating_col):
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix

def build_matrix_factorization_model(num_clients, num_products, embedding_dim=25, l2_reg=1e-6, dropout_rate=0.5):
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

def train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def generate_recommendations(client_encoder, specific_client_id, model, num_products, bought_products):
    specific_client_encoded_id = client_encoder.transform([specific_client_id])[0]
    client_product_combinations = np.array([(specific_client_encoded_id, product) for product in range(num_products) if product not in bought_products])

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

def compute_recall_at_k(recommendations, true_bought_products, k):
    top_k_recommendations = recommendations.head(k)['Product_ID'].tolist()
    hits = len(set(top_k_recommendations) & set(true_bought_products))
    return hits / len(true_bought_products)

def main():
    data_path = "/Users/igormol/Desktop/analytics3/sample/Marajo.csv"
    specific_client_original_id = "MASTELARI TRANSPORTE LTDA"

    data = load_data(data_path)
    data, client_encoder, product_encoder = encode_ids(data)
    data = aggregate_duplicates(data, 'Client_ID', 'Product_ID', 'Monetary_Value')

    interaction_matrix = create_interaction_matrix(data, 'Client_ID', 'Product_ID', 'Monetary_Value')
    interaction_matrix_values = interaction_matrix.values

    num_clients, num_products = interaction_matrix.shape

    X = np.argwhere(interaction_matrix_values)
    y = interaction_matrix_values[X[:, 0], X[:, 1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mf_model = build_matrix_factorization_model(num_clients, num_products)
    train_matrix_factorization_model(mf_model, X_train, y_train)

    client_encoded_id = client_encoder.transform([specific_client_original_id])[0]
    bought_products = data[data['Client_ID'] == client_encoded_id]['Product_ID'].unique()

    all_predictions = generate_recommendations(client_encoder, specific_client_original_id, mf_model, num_products, bought_products)
    all_predictions = all_predictions.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID', how='left')
    all_predictions = all_predictions.sort_values(by='Predicted_Rating', ascending=False)

    print(all_predictions)

    # Save the top 15 recommendations to a CSV file
    top_recommendations = all_predictions.nlargest(15, 'Predicted_Rating')
    top_recommendations.to_csv('top_recommendations.csv', index=False)

    # Compute the recall at 15
    recall_at_15 = compute_recall_at_k(all_predictions, bought_products, 15)
    print(f'Recall at 15: {recall_at_15}')

if __name__ == "__main__":
    main()
