# hybrid_recsys_web.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains the functions related to the hybrid recommender
# system's model. The first phase utilizes an autoencoder neural network
# trained on the user-item interaction matrix. In the second phase, the encoded
# vectors predicted by the autoencoder are used as input to a gradient boosting
# machine, the results of which are then decoded to generate the vectors of
# predicted ratings for each user.

# In Section 1, we define the functions responsible for loading and preprocessing
# the user data and for aggregating data for the autoencoder. In Section 2, we
# define the routines responsible for the autoencoder neural network and the
# gradient boosting machine, including both the functions for building the
# architecture and the routines for training the networks. In Section 3, we
# write the algorithms that generate the recommendation tables and orchestrate
# the main routine.

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import xgboost as xgb

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
    autoencoder.fit(train_data, train_data, epochs=10, batch_size=128, shuffle=True, validation_data=(test_data, test_data), verbose=1)

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

# Section 3: Principal routines; web interface.

def run_hybrid_recsys(data):
    # Define a function named "run_hybrid_recsys" that takes one parameter "data"

    # Set the title of the Streamlit app to "Hybrid Recommendation System"
    st.title("Hybrid Recommendation System")

    # Display a brief description for the user to upload a CSV file and choose a client for recommendations
    st.write("Upload your CSV file and choose a client to get recommendations.")

    # Preprocess the data and encode user and item identifiers
    data, user_encoder, item_encoder = hybrid_recsys_preprocess_data(data)

    # Get the unique client names from the data
    client_names = data['Client_ID'].unique()

    # Display a dropdown menu for selecting a client, with an initial option to "Select a client"
    client_name = st.selectbox("Select a client", ["Select a client"] + list(client_names))

    # If a valid client is selected (not the default "Select a client" option)
    if client_name and client_name != "Select a client":
        # Aggregate the data by summing the 'Monetary_Value' for each user-item pair
        data_aggregated = hybrid_recsys_aggregate_data(data)

        # Create an interaction matrix from the aggregated data
        interaction_matrix = hybrid_recsys_create_interaction_matrix(data_aggregated)

        # Split the interaction matrix into training and testing sets
        train_data, test_data = hybrid_recsys_train_test_split(interaction_matrix)

        # Get the number of items (input dimension for the autoencoder)
        input_dim = interaction_matrix.shape[1]

        # Train the autoencoder using the training and testing data
        encoded_train_data, encoder = hybrid_recsys_train_autoencoder(train_data, test_data, input_dim)

        # Train the XGBoost model using the encoded training data
        model = hybrid_recsys_train_xgboost(encoded_train_data, train_data)

        # Generate the top 10 recommendations for the selected client
        top_10_recommendations = hybrid_recsys_generate_recommendations(client_name, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix, data)

        # Display a message with the top 10 recommendations for the selected client
        st.write("Top 10 Recommendations for ", client_name)

        # Display the top 10 recommendations in a table with Product_ID, Product_Name, and predicted_rating columns
        st.table(top_10_recommendations[['Product_ID', 'Product_Name', 'predicted_rating']])

def main():
    # Define a function named "main" that serves as the entry point of the Streamlit app

    # Set the title of the sidebar to "Navigation"
    st.sidebar.title("Navigation")

    # Create a file uploader widget in the sidebar for uploading CSV files
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    # If a file has been uploaded
    if uploaded_file is not None:
        # Load the data from the uploaded CSV file using the hybrid_recsys_load_data function
        data = hybrid_recsys_load_data(uploaded_file)

    # Create a select box in the sidebar for choosing between "Home" and "Recommender System" tabs
    tab = st.sidebar.selectbox("Choose a tab", ["Home", "Recommender System"])

    # If the "Home" tab is selected
    if tab == "Home":
        # Set the title of the main page to "Home"
        st.title("Home")

        # Display a message in the main page indicating that this is the home tab
        st.write("This is the home tab.")

    # If the "Recommender System" tab is selected
    elif tab == "Recommender System":
        # If a file has been uploaded
        if uploaded_file is not None:
            # Run the hybrid recommendation system with the uploaded data
            run_hybrid_recsys(data)
        else:
            # Display a message asking the user to upload a CSV file to use the recommender system
            st.write("Please upload a CSV file to use the recommender system.")

if __name__ == "__main__":
    main()
