import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

# Load and preprocess data
def recsys_products_load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    client_name_map = data[['Client_ID']].drop_duplicates().reset_index(drop=True)
    client_name_map['Client_Name'] = client_name_map['Client_ID']

    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])

    return data, client_name_map, client_encoder, product_encoder

# Split data
def recsys_products_split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

# Convert date to numerical format
def recsys_products_convert_date_to_numerical(data):
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date']).astype(int) / 10**9
    return data

# Define and train autoencoder
def train_autoencoder(train_data):
    input_dim = train_data.shape[1]
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

    encoder = Model(input_layer, encoded)
    return encoder

# Extract features using autoencoder
def extract_features_with_autoencoder(encoder, data):
    return encoder.predict(data)

# Train model
def recsys_products_train_model(train, train_features, target):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(train_features, train[target])
    return model

# Predict and evaluate
def recsys_products_predict_and_evaluate(model, test, test_features, target):
    test['Predicted_Value'] = model.predict(test_features)
    mse = mean_squared_error(test[target], test['Predicted_Value'])
    return test, mse

# Normalize scores
def recsys_products_normalize_scores(df):
    min_score = df['Predicted_Value'].min()
    max_score = df['Predicted_Value'].max()
    df['Normalized_Score'] = 100 * (df['Predicted_Value'] - min_score) / (max_score - min_score)
    return df

# Compute recall at k
def recsys_products_recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))

# Generate recommendations
def recsys_products_generate_recommendations(test, top_k):
    recommendations = defaultdict(list)
    clients = test['Client_ID'].unique()
    for client in clients:
        client_data = test[test['Client_ID'] == client]
        client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False)
        unique_recommendations = client_recommendations.drop_duplicates(subset='Product_ID').head(top_k)
        recommendations[client] = list(unique_recommendations['Product_ID'])
    return recommendations

# Calculate recall scores
def recsys_products_calculate_recall_scores(test, recommendations, top_k):
    recall_scores = []
    clients = test['Client_ID'].unique()
    for client in clients:
        actual_products = test[test['Client_ID'] == client]['Product_ID']
        recall = recsys_products_recall_at_k(actual_products, recommendations[client], top_k)
        recall_scores.append(recall)
    return np.mean(recall_scores)

# Map product IDs to names
def recsys_products_map_product_ids_to_names(data):
    return dict(zip(data['Product_ID'], data['Product_Name']))

# Decode client IDs to names
def recsys_products_decode_client_ids_to_names(client_name_map, client_encoder):
    client_name_map['Client_ID'] = client_encoder.transform(client_name_map['Client_Name'])
    return client_name_map

# Main function
def main():
    st.title("Product Recommendation System")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data, client_name_map, client_encoder, product_encoder = recsys_products_load_and_preprocess_data(uploaded_file)
        train, test = recsys_products_split_data(data)

        train = recsys_products_convert_date_to_numerical(train)
        test = recsys_products_convert_date_to_numerical(test)

        features = ['Sales_Date', 'Client_ID', 'Product_ID']
        target = 'Monetary_Value'

        # Train autoencoder and extract features
        encoder = train_autoencoder(train[features])
        train_autoencoded_features = extract_features_with_autoencoder(encoder, train[features])
        test_autoencoded_features = extract_features_with_autoencoder(encoder, test[features])

        # Convert to DataFrame to match train and test indices
        train_autoencoded_df = pd.DataFrame(train_autoencoded_features, index=train.index)
        test_autoencoded_df = pd.DataFrame(test_autoencoded_features, index=test.index)

        # Add autoencoded features to the original features
        train_combined = pd.concat([train[features], train_autoencoded_df], axis=1)
        test_combined = pd.concat([test[features], test_autoencoded_df], axis=1)

        model = recsys_products_train_model(train, train_combined, target)
        test, mse = recsys_products_predict_and_evaluate(model, test, test_combined, target)
        st.write(f'Mean Squared Error: {mse}')

        test = recsys_products_normalize_scores(test)

        top_k = 10
        recommendations = recsys_products_generate_recommendations(test, top_k)

        average_recall = recsys_products_calculate_recall_scores(test, recommendations, top_k)
        st.write(f'Average Recall at {top_k}: {average_recall}')

        product_map = recsys_products_map_product_ids_to_names(data)
        client_name_map = recsys_products_decode_client_ids_to_names(client_name_map, client_encoder)

        clients = test['Client_ID'].unique()

        # Default to showing all clients
        client_name = st.selectbox('Select a Client', ['All Clients'] + list(client_name_map['Client_Name'].unique()))

        if client_name == 'All Clients':
            for client in clients:
                client_name = client_name_map[client_name_map['Client_ID'] == client]['Client_Name'].values[0]
                st.subheader(f'Recommendations for {client_name}')
                client_data = test[test['Client_ID'] == client]
                client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False).drop_duplicates(subset='Product_ID').head(top_k)
                recommendations_list = []
                for _, row in client_recommendations.iterrows():
                    recommendations_list.append({
                        "Product Name": product_map[row['Product_ID']],
                        "Predicted Score": row['Normalized_Score']
                    })
                st.table(pd.DataFrame(recommendations_list))
        else:
            client_id = client_name_map[client_name_map['Client_Name'] == client_name]['Client_ID'].values[0]
            client_data = test[test['Client_ID'] == client_id]
            client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False).drop_duplicates(subset='Product_ID').head(top_k)
            recommendations_list = []
            for _, row in client_recommendations.iterrows():
                recommendations_list.append({
                    "Product Name": product_map[row['Product_ID']],
                    "Predicted Score": row['Normalized_Score']
                })
            st.table(pd.DataFrame(recommendations_list))

# Run the main function
if __name__ == "__main__":
    main()
