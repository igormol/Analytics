# recsys_products_clients.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import streamlit as st
from sklearn.metrics import mean_squared_error  # Import mean_squared_error

# Function to load data from a CSV file and handle missing Brand_Name values
def recsys_products_to_clients_load_data(path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(path)
    # Fill missing Brand_Name values with 'Unknown'
    data['Brand_Name'].fillna('Unknown', inplace=True)
    return data  # Return the DataFrame

# Function to encode Client_ID and Product_ID using LabelEncoder
def recsys_products_to_clients_encode_ids(data):
    # Initialize LabelEncoders for Client_ID and Product_ID
    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    # Fit and transform Client_ID and Product_ID columns
    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])
    return data, client_encoder, product_encoder  # Return encoded data and encoders

# Function to aggregate duplicate entries by summing rating values
def recsys_products_to_clients_aggregate_duplicates(data, client_col, product_col, rating_col):
    # Group data by client and product, summing the ratings
    aggregated_data = data.groupby([client_col, product_col], as_index=False)[rating_col].sum()
    # Merge with unique Product_ID and Product_Name pairs
    aggregated_data = aggregated_data.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on=product_col, how='left')
    return aggregated_data  # Return the aggregated data

# Function to create an interaction matrix (client-product rating matrix)
def recsys_products_to_clients_create_interaction_matrix(data, client_col, product_col, rating_col):
    # Create a pivot table with clients as rows, products as columns, and ratings as values
    interaction_matrix = data.pivot(index=client_col, columns=product_col, values=rating_col).fillna(0)
    return interaction_matrix  # Return the interaction matrix

# Function to build a matrix factorization model using TensorFlow and Keras
def recsys_products_to_clients_build_matrix_factorization_model(num_clients, num_products, embedding_dim=25, l2_reg=1e-6, dropout_rate=0.5):
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

# Function to train the matrix factorization model
def recsys_products_to_clients_train_matrix_factorization_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    # Train the model on the training data
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Function to generate product recommendations for a specific client
def recsys_products_to_clients_generate_recommendations(client_encoder, specific_client_id, model, num_products):
    # Encode the specific client ID
    specific_client_encoded_id = client_encoder.transform([specific_client_id])[0]
    # Create all possible client-product pairs for the specific client
    client_product_combinations = np.array([(specific_client_encoded_id, product) for product in range(num_products)])

    # Extract client and product IDs for prediction
    all_client_ids = client_product_combinations[:, 0]
    all_product_ids = client_product_combinations[:, 1]
    # Predict ratings for all client-product pairs
    predictions = model.predict([all_client_ids, all_product_ids]).flatten()

    # Normalize the predicted ratings to a 0-100 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    predictions_normalized = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

    # Return a DataFrame with the client ID, product IDs, and normalized predicted ratings
    return pd.DataFrame({
        'Client_ID': [specific_client_id] * len(predictions_normalized),
        'Product_ID': all_product_ids,
        'Predicted_Rating': predictions_normalized
    })

# Function to compute MSE for all recommendations
def compute_mse_for_recommendations(model, X, y):
    y_pred = model.predict([X[:, 0], X[:, 1]]).flatten()
    mse = mean_squared_error(y, y_pred)
    return mse
