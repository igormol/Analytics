# recsys_clients_to_products.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Loads data from a CSV file and process dates
def recsys_clients_to_products_load_data(path):
    # Reset the file pointer to the start of the file
    path.seek(0)
    # Read the CSV file into a DataFrame
    data = pd.read_csv(path)
    # Convert Sales_Date to datetime, handling errors by coercing to NaT
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date'], errors='coerce')
    # Drop rows with NaT in Sales_Date
    data.dropna(subset=['Sales_Date'], inplace=True)
    # Calculate Recency as the number of days since the sales date
    data['Recency'] = (datetime.now() - data['Sales_Date']).dt.days
    return data  # Return the processed DataFrame

# Encodes Product_ID using LabelEncoder
def recsys_clients_to_products_encode_product_ids(data):
    # Initialize LabelEncoder for Product_ID
    product_encoder = LabelEncoder()
    # Fit and transform Product_Name to create Product_ID
    data['Product_ID'] = product_encoder.fit_transform(data['Product_Name'])
    return data, product_encoder  # Return encoded data and the encoder

# Aggregates duplicate entries by summing rating values
def recsys_clients_to_products_aggregate_duplicates(data, client_col, product_col, rating_col):
    # Group data by client and product, summing the ratings
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    # Merge with unique Product_ID and Product_Name pairs
    aggregated_data = aggregated_data.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on=product_col, how='left')
    return aggregated_data  # Return the aggregated data

# Creates an interaction matrix (client-product rating matrix)
def recsys_clients_to_products_create_interaction_matrix(data, client_col, product_col, rating_col):
    # Create a pivot table with clients as rows, products as columns, and ratings as values
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix  # Return the interaction matrix

# Builds a matrix factorization model using TensorFlow and Keras
def recsys_clients_to_products_build_matrix_factorization_model(num_clients, num_products, embedding_dim=25, l2_reg=1e-6, dropout_rate=0.5):
    # Define input for client IDs
    client_input = keras.layers.Input(shape=(1,), name='client_input')
    # Embed client IDs into dense vectors
    client_embedding = keras.layers.Embedding(input_dim=num_clients, output_dim=embedding_dim,
                                              embeddings_regularizer=keras.regularizers.l2(l2_reg),
                                              name='client_embedding')(client_input)
    # Flatten the client embedding
    client_vec = keras.layers.Flatten()(client_embedding)

    # Define input for product IDs
    product_input = keras.layers.Input(shape=(1,), name='product_input')
    # Embed product IDs into dense vectors
    product_embedding = keras.layers.Embedding(input_dim=num_products, output_dim=embedding_dim,
                                               embeddings_regularizer=keras.regularizers.l2(l2_reg),
                                               name='product_embedding')(product_input)
    # Flatten the product embedding
    product_vec = keras.layers.Flatten()(product_embedding)

    # Concatenate client and product vectors
    concatenated = keras.layers.Concatenate()([client_vec, product_vec])
    # Apply dropout for regularization
    dropout = keras.layers.Dropout(dropout_rate)(concatenated)
    # Define the output layer
    output = keras.layers.Dense(1)(dropout)

    # Build and compile the model
    model = keras.models.Model(inputs=[client_input, product_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model  # Return the compiled model

# Trains the matrix factorization model
def recsys_clients_to_products_train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    # Train the model on the training data
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Generates client recommendations for a specific product
def recsys_clients_to_products_generate_client_recommendations(product_encoder, specific_product_name, model, all_client_ids):
    # Encode the specific product name
    specific_product_encoded_id = product_encoder.transform([specific_product_name])[0]
    # Create all possible client-product pairs for the specific product
    product_client_combinations = np.array([(i, specific_product_encoded_id) for i in range(len(all_client_ids))])

    # Extract client and product IDs for prediction
    client_ids_encoded = product_client_combinations[:, 0]
    product_ids = product_client_combinations[:, 1]
    # Predict ratings for all client-product pairs
    predictions = model.predict([client_ids_encoded, product_ids]).flatten()

    # Normalize the predicted ratings to a 0-100 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

    # Return a DataFrame with the product name, encoded and original client IDs, and normalized predicted ratings
    return pd.DataFrame({
        'Product_Name': [specific_product_name] * len(predictions_normalized),
        'Client_ID_Encoded': client_ids_encoded,
        'Client_ID': all_client_ids,
        'Predicted_Rating': predictions_normalized
    })
