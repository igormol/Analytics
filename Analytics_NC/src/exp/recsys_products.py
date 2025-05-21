import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import defaultdict
from tabulate import tabulate

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    client_name_map = data[['Client_ID']].drop_duplicates().reset_index(drop=True)
    client_name_map['Client_Name'] = client_name_map['Client_ID']

    client_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    data['Client_ID'] = client_encoder.fit_transform(data['Client_ID'])
    data['Product_ID'] = product_encoder.fit_transform(data['Product_ID'])

    return data, client_name_map, client_encoder, product_encoder

# Split data
def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

# Convert date to numerical format
def convert_date_to_numerical(data):
    data['Sales_Date'] = pd.to_datetime(data['Sales_Date']).astype(int) / 10**9
    return data

# Train model
def train_model(train, features, target):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(train[features], train[target])
    return model

# Predict and evaluate
def predict_and_evaluate(model, test, features, target):
    test['Predicted_Value'] = model.predict(test[features])
    mse = mean_squared_error(test[target], test['Predicted_Value'])
    return test, mse

# Normalize scores
def normalize_scores(df):
    min_score = df['Predicted_Value'].min()
    max_score = df['Predicted_Value'].max()
    df['Normalized_Score'] = 100 * (df['Predicted_Value'] - min_score) / (max_score - min_score)
    return df

# Compute recall at k
def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))

# Generate recommendations
def generate_recommendations(test, top_k):
    recommendations = defaultdict(list)
    clients = test['Client_ID'].unique()
    for client in clients:
        client_data = test[test['Client_ID'] == client]
        client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False)
        unique_recommendations = client_recommendations.drop_duplicates(subset='Product_ID').head(top_k)
        recommendations[client] = list(unique_recommendations['Product_ID'])
    return recommendations

# Calculate recall scores
def calculate_recall_scores(test, recommendations, top_k):
    recall_scores = []
    clients = test['Client_ID'].unique()
    for client in clients:
        actual_products = test[test['Client_ID'] == client]['Product_ID']
        recall = recall_at_k(actual_products, recommendations[client], top_k)
        recall_scores.append(recall)
    return np.mean(recall_scores)

# Map product IDs to names
def map_product_ids_to_names(data):
    return dict(zip(data['Product_ID'], data['Product_Name']))

# Decode client IDs to names
def decode_client_ids_to_names(client_name_map, client_encoder):
    client_name_map['Client_ID'] = client_encoder.transform(client_name_map['Client_Name'])
    return client_name_map

# Save recommendations to file
def save_recommendations_to_file(clients, client_name_map, recommendations, product_map, test, top_k, file_name="recommendations.txt"):
    with open(file_name, 'w') as f:
        for client in clients:
            client_name = client_name_map[client_name_map['Client_ID'] == client]['Client_Name'].values[0]
            f.write(f'Client Name: {client_name}\n')
            f.write('Top 10 Recommendations:\n')
            client_data = test[test['Client_ID'] == client]
            client_recommendations = client_data.sort_values(by='Normalized_Score', ascending=False)
            unique_recommendations = client_recommendations.drop_duplicates(subset='Product_ID').head(top_k)
            recommendations_list = []
            for _, row in unique_recommendations.iterrows():
                product_id = row['Product_ID']
                recommendations_list.append([product_map[product_id], row['Normalized_Score']])
            f.write(tabulate(recommendations_list, headers=['Product Name', 'Predicted Score'], tablefmt='grid'))
            f.write('\n\n')

# Main function
def main(file_path):
    data, client_name_map, client_encoder, product_encoder = load_and_preprocess_data(file_path)
    train, test = split_data(data)

    train = convert_date_to_numerical(train)
    test = convert_date_to_numerical(test)

    features = ['Sales_Date', 'Client_ID', 'Product_ID']
    target = 'Monetary_Value'

    model = train_model(train, features, target)
    test, mse = predict_and_evaluate(model, test, features, target)
    print(f'Mean Squared Error: {mse}')

    test = normalize_scores(test)

    top_k = 10
    recommendations = generate_recommendations(test, top_k)

    average_recall = calculate_recall_scores(test, recommendations, top_k)
    print(f'Average Recall at {top_k}: {average_recall}')

    product_map = map_product_ids_to_names(data)
    client_name_map = decode_client_ids_to_names(client_name_map, client_encoder)

    clients = test['Client_ID'].unique()
    save_recommendations_to_file(clients, client_name_map, recommendations, product_map, test, top_k)

# Run the main function
file_path = "/Users/igormol/Desktop/Analytics_NC/sample/CPP.csv"
main(file_path)
