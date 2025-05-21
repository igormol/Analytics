import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # Import mean_squared_error

# Section 1: Data loading and preprocessing

def hybrid_recsys_load_data(file):
    # Use pandas to read the CSV file located at the path specified by "file"
    data = pd.read_csv(file)

    # Return the DataFrame containing the data read from the CSV file
    return data

def hybrid_recsys_preprocess_data(data):
    # Fill missing values in the 'Brand_Name' column with empty strings
    data['Brand_Name'] = data['Brand_Name'].fillna('')

    # Create an instance of LabelEncoder for encoding user IDs
    user_encoder = LabelEncoder()

    # Create an instance of LabelEncoder for encoding item IDs
    item_encoder = LabelEncoder()

    # Transform 'Client_ID' column values to numerical user IDs and store them in 'user_id' column
    data['user_id'] = user_encoder.fit_transform(data['Client_ID'])

    # Transform 'Product_ID' column values to numerical item IDs and store them in 'item_id' column
    data['item_id'] = item_encoder.fit_transform(data['Product_ID'])

    # Return the processed data along with the user and item encoders
    return data, user_encoder, item_encoder

def hybrid_recsys_aggregate_data(data):
    # Group the data by 'user_id' and 'item_id' and sum the 'Monetary_Value' for each group
    data_aggregated = data.groupby(['user_id', 'item_id'], as_index=False)['Monetary_Value'].sum()

    # Return the aggregated data
    return data_aggregated

def hybrid_recsys_create_interaction_matrix(data_aggregated):
    # Create an interaction matrix by pivoting the data, with 'user_id' as rows, 'item_id' as columns, and 'Monetary_Value' as values
    # Fill missing values with 0
    interaction_matrix = data_aggregated.pivot(index='user_id', columns='item_id', values='Monetary_Value').fillna(0)

    # Return the interaction matrix
    return interaction_matrix

# Section 2: Machine Learning Routines

def hybrid_recsys_train_test_split(interaction_matrix):
    # Split the interaction matrix into training and testing sets, with 20% of the data in the test set and a fixed random state for reproducibility
    train_data, test_data = train_test_split(interaction_matrix, test_size=0.2, random_state=42)

    # Return the training and testing sets
    return train_data, test_data

def hybrid_recsys_train_autoencoder(train_data, test_data, input_dim, encoding_dim=10):
    # Define the input layer for the autoencoder with the shape equal to input_dim
    input_layer = Input(shape=(input_dim,))

    # Add an encoding layer to the autoencoder with the specified encoding_dim and ReLU activation function
    encoded = Dense(encoding_dim, activation='relu')(input_layer)

    # Add a decoding layer to the autoencoder with the shape equal to input_dim and sigmoid activation function
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Define the autoencoder model with the input layer and decoded output
    autoencoder = Model(input_layer, decoded)

    # Compile the autoencoder model with Adam optimizer and mean squared error loss function
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder model using the training data for both input and output, with specified epochs, batch size, and validation data
    autoencoder.fit(train_data, train_data, epochs=10, batch_size=256, shuffle=True, validation_data=(test_data, test_data), verbose=1)

    # Define the encoder model to extract the encoded representation from the input layer
    encoder = Model(input_layer, encoded)

    # Use the encoder to transform the training data into the encoded representation
    encoded_train_data = encoder.predict(train_data)

    # Return the encoded training data and the encoder model
    return encoded_train_data, encoder

def hybrid_recsys_train_xgboost(encoded_train_data, train_data):
    # Set the feature matrix (input data) for the XGBoost model as the encoded training data
    x_train = encoded_train_data

    # Set the target matrix (output data) for the XGBoost model as the values from the original training data
    y_train = train_data.values

    # Reshape y_train if it is a 1-dimensional array to ensure it is a 2-dimensional array
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    # Initialize the XGBoost regressor model with specified parameters
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=5, max_depth=5, learning_rate=0.5)

    # Train the XGBoost model using the feature matrix and target matrix
    model.fit(x_train, y_train)

    # Return the trained XGBoost model
    return model

def hybrid_recsys_generate_recommendations(user_name, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix, data):
    # Transform the user name to the corresponding user_id using the user_encoder
    user_id = user_encoder.transform([user_name])[0]

    # Get the encoded representation of the user's interaction vector
    user_vector = encoded_train_data[user_id].reshape(1, -1)

    # Use the trained model to predict the ratings for all items for this user
    predicted_ratings = model.predict(user_vector).flatten()

    # Ensure the length of the predicted ratings matches the number of items in the interaction matrix
    assert len(predicted_ratings) == interaction_matrix.shape[1], "Predicted ratings length mismatch."

    # Normalize the predicted ratings to a scale of 0-100
    min_rating = predicted_ratings.min()
    max_rating = predicted_ratings.max()
    normalized_ratings = 100 * (predicted_ratings - min_rating) / (max_rating - min_rating)

    # Create a DataFrame with item_ids and their corresponding normalized predicted ratings
    recommendations = pd.DataFrame({'item_id': range(len(normalized_ratings)), 'predicted_rating': normalized_ratings})

    # Sort the recommendations by predicted rating in descending order and select the top 10 items
    top_10_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(10)

    # Transform the item_ids back to the original Product_IDs using the item_encoder
    top_10_recommendations['Product_ID'] = item_encoder.inverse_transform(top_10_recommendations['item_id'])

    # Merge the top 10 recommendations with the original data to get the product names
    top_10_recommendations = top_10_recommendations.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID')

    # Return the top 10 recommendations with product names
    return top_10_recommendations

# Section 4: Metrics

def compute_recall_at_k(client_name, recommendations, test_data, user_encoder, item_encoder, k=10):
    user_id = user_encoder.transform([client_name])[0]
    true_items = test_data.columns[test_data.loc[user_id] > 0]
    recommended_items = item_encoder.transform(recommendations['Product_ID'])
    true_items_set = set(true_items)
    recommended_items_set = set(recommended_items[:k])
    hits = len(true_items_set & recommended_items_set)
    recall = hits / len(true_items_set) if len(true_items_set) > 0 else 0
    return recall

def compute_mse(client_name, recommendations, test_data, user_encoder, item_encoder):
    user_id = user_encoder.transform([client_name])[0]
    true_ratings = test_data.loc[user_id, item_encoder.transform(recommendations['Product_ID'])]
    predicted_ratings = recommendations['predicted_rating']
    mse = mean_squared_error(true_ratings, predicted_ratings)
    return mse

def compute_avg_train_mse(client_names, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix):
    mse_list = []
    max_user_id = encoded_train_data.shape[0]

    for client_name in client_names:
        try:
            user_id = user_encoder.transform([client_name])[0]
            if user_id >= max_user_id:
                # Skip user_ids that are out of bounds
                continue
            user_vector = encoded_train_data[user_id].reshape(1, -1)
            predicted_ratings = model.predict(user_vector).flatten()
            true_ratings = interaction_matrix.iloc[user_id].values
            mse = mean_squared_error(true_ratings, predicted_ratings)
            mse_list.append(mse)
        except IndexError as e:
            print(f"IndexError for client {client_name} with user_id {user_id}: {e}")
        except Exception as e:
            print(f"Unexpected error for client {client_name} with user_id {user_id}: {e}")

    if mse_list:
        avg_mse = sum(mse_list) / len(mse_list)
    else:
        avg_mse = float('nan')  # Handle the case where mse_list is empty

    return avg_mse
