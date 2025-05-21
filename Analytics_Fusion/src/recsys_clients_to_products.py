# recsys_clients_to_products.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # Import mean_squared_error

def recsys_clients_to_products_load_data(file):
    # Load the data from a CSV file into a DataFrame.
    data = pd.read_csv(file)
    # Return the loaded data.
    return data

def recsys_clients_to_products_preprocess_data(data):
    # Fill missing values in the 'Brand_Name' column with an empty string.
    data['Brand_Name'] = data['Brand_Name'].fillna('')
    # Initialize a LabelEncoder for encoding client IDs.
    user_encoder = LabelEncoder()
    # Initialize a LabelEncoder for encoding product IDs.
    item_encoder = LabelEncoder()
    # Encode the 'Client_ID' column to numerical user IDs.
    data['user_id'] = user_encoder.fit_transform(data['Client_ID'])
    # Encode the 'Product_ID' column to numerical item IDs.
    data['item_id'] = item_encoder.fit_transform(data['Product_ID'])
    # Return the preprocessed data along with the user and item encoders.
    return data, user_encoder, item_encoder

def recsys_clients_to_products_aggregate_data(data):
    # Aggregate the monetary value for each (user_id, item_id) pair.
    data_aggregated = data.groupby(['user_id', 'item_id'], as_index=False)['Monetary_Value'].sum()
    # Return the aggregated data.
    return data_aggregated

def recsys_clients_to_products_create_interaction_matrix(data_aggregated):
    # Create an interaction matrix with item IDs as rows and user IDs as columns,
    # and fill missing values with 0.
    interaction_matrix = data_aggregated.pivot(index='item_id', columns='user_id', values='Monetary_Value').fillna(0)
    # Return the interaction matrix.
    return interaction_matrix

def recsys_clients_to_products_train_test_split(interaction_matrix):
    # Split the interaction matrix into training and test datasets, with 20% of the data as the test set.
    train_data, test_data = train_test_split(interaction_matrix, test_size=0.2, random_state=42)
    # Return the training and test datasets.
    return train_data, test_data

def recsys_clients_to_products_train_autoencoder(train_data, test_data, input_dim, encoding_dim=10):
    # Define the input layer with the shape equal to the number of features (input_dim).
    input_layer = Input(shape=(input_dim,))

    # Create an encoding layer with the specified encoding dimension and ReLU activation function.
    encoded = Dense(encoding_dim, activation='relu')(input_layer)

    # Create a decoding layer with the original input dimension and sigmoid activation function.
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Define the autoencoder model that maps the input layer to the decoded output.
    autoencoder = Model(input_layer, decoded)

    # Compile the autoencoder model with Adam optimizer and mean squared error loss function.
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder on the training data, using the training data as both input and output.
    autoencoder.fit(
        train_data, train_data,
        epochs=10, batch_size=128,
        shuffle=True, validation_data=(test_data, test_data),
        verbose=1
    )

    # Define the encoder model that maps the input layer to the encoded representation.
    encoder = Model(input_layer, encoded)

    # Encode the training data using the trained encoder.
    encoded_train_data = encoder.predict(train_data)

    # Return the encoded training data and the encoder model.
    return encoded_train_data, encoder

def recsys_clients_to_products_train_xgboost(encoded_train_data, train_data):
    # Assign the encoded training data to x_train.
    x_train = encoded_train_data

    # Extract the values from the training data as the target variable y_train.
    y_train = train_data.values

    # Ensure y_train is a 2D array.
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    # Initialize an XGBoost regressor with specified hyperparameters.
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=5,
        max_depth=5,
        learning_rate=0.5
    )

    # Fit the XGBoost model on the encoded training data.
    model.fit(x_train, y_train)

    # Return the trained XGBoost model.
    return model

def recsys_clients_to_products_generate_recommendations(product_name, item_encoder, encoded_train_data, model, user_encoder, interaction_matrix, data):
    # Get the product ID for the given product name.
    product_id = data[data['Product_Name'] == product_name]['Product_ID'].iloc[0]

    # Encode the product ID using the item encoder.
    product_id_encoded = item_encoder.transform([product_id])[0]

    # Get the encoded vector for the specified product.
    product_vector = encoded_train_data[product_id_encoded].reshape(1, -1)

    # Predict the ratings for all users using the XGBoost model.
    predicted_ratings = model.predict(product_vector).flatten()

    # Ensure the length of predicted ratings matches the number of users in the interaction matrix.
    assert len(predicted_ratings) == interaction_matrix.shape[1], "Predicted ratings length mismatch."

    # Normalize the predicted ratings to a 0-100 scale.
    min_rating = predicted_ratings.min()
    max_rating = predicted_ratings.max()
    normalized_ratings = 100 * (predicted_ratings - min_rating) / (max_rating - min_rating)

    # Create a DataFrame of user IDs and their normalized predicted ratings.
    recommendations = pd.DataFrame({'user_id': range(len(normalized_ratings)), 'predicted_rating': normalized_ratings})

    # Get the top 10 recommendations sorted by predicted rating in descending order.
    top_10_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(10)

    # Decode the user IDs back to their original client IDs.
    top_10_recommendations['Client_ID'] = user_encoder.inverse_transform(top_10_recommendations['user_id'])

    # Calculate the MSE for each recommendation.
    mse_per_user = []
    for user_id in top_10_recommendations['user_id']:
        actual_values = interaction_matrix.iloc[:, user_id].values
        predicted_values = interaction_matrix.values.dot(predicted_ratings).flatten()
        mse = mean_squared_error(actual_values, predicted_values)
        mse_per_user.append(mse)

    mse_df = pd.DataFrame({'user_id': top_10_recommendations['user_id'], 'MSE': mse_per_user})

    # Calculate the mean MSE for all recommendations.
    mean_mse = mse_df['MSE'].mean()

    # Return the top 10 recommendations with client IDs and predicted ratings, and the MSE values.
    return top_10_recommendations, mse_df, mean_mse
