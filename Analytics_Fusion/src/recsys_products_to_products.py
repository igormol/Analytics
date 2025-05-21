# recsys_products_to_products.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st

def recsys_products_to_products_load_data(file):
    # Load data from a CSV file into a DataFrame.
    df = pd.read_csv(file)

    # Convert the 'Sales_Date' column to datetime format, handling errors by setting them as NaT.
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'], errors='coerce')

    # Convert the 'Monetary_Value' column to numeric format, handling errors by setting them as NaN.
    df['Monetary_Value'] = pd.to_numeric(df['Monetary_Value'], errors='coerce')

    # Convert the 'Discount' column to numeric format, handling errors by setting them as NaN.
    df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')

    # Fill any remaining missing values in the DataFrame with empty strings.
    df.fillna('', inplace=True)

    # Return the cleaned DataFrame.
    return df

def recsys_products_to_products_create_user_item_matrix(df):
    # Create a user-item matrix where rows are clients and columns are products,
    # with values representing the sum of monetary values for each product bought by each client.
    user_item_matrix = df.pivot_table(
        index='Client_ID',
        columns='Product_Name',
        values='Monetary_Value',
        aggfunc='sum',
        fill_value=0
    )

    # Return the user-item matrix.
    return user_item_matrix

def recsys_products_to_products_fit_knn_model(matrix):
    # Initialize a K-Nearest Neighbors model with cosine similarity as the distance metric
    # and using the brute-force algorithm for finding neighbors.
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the K-Nearest Neighbors model to the user-item matrix.
    model_knn.fit(matrix)

    # Return the fitted K-Nearest Neighbors model.
    return model_knn

def recsys_products_to_products_get_top_n_recommendations(product_name, product_user_matrix, model_knn, n=10):
    # Check if the given product name is in the product-user matrix.
    if product_name not in product_user_matrix.index:
        return []

    # Get the vector of the specified product from the product-user matrix.
    product_vector = product_user_matrix.loc[[product_name]].values

    # Find the n+1 nearest neighbors (including the product itself) using the K-Nearest Neighbors model.
    distances, indices = model_knn.kneighbors(product_vector, n_neighbors=n+1)

    # Create a list of recommended products and their corresponding distances, excluding the product itself.
    recommended_products = [
        (product_user_matrix.index[i], distances.flatten()[j])
        for j, i in enumerate(indices.flatten())
        if product_user_matrix.index[i] != product_name
    ]

    # Return the top n recommended products.
    return recommended_products[:n]
