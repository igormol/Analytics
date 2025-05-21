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

# Generate recommendations
def generate_recommendations_reverse(test, top_k):
    recommendations = defaultdict(list)
    products = test['Product_ID'].unique()
    for product in products:
        product_data = test[test['Product_ID'] == product]
        product_recommendations = product_data.sort_values(by='Normalized_Score', ascending=False)
        unique_recommendations = product_recommendations.drop_duplicates(subset='Client_ID').head(top_k)
        recommendations[product] = list(unique_recommendations['Client_ID'])
    return recommendations

# Map product IDs to names
def map_product_ids_to_names(data):
    return dict(zip(data['Product_ID'], data['Product_Name']))

# Decode client IDs to names
def decode_client_ids_to_names(client_name_map, client_encoder):
    client_name_map['Client_ID'] = client_encoder.transform(client_name_map['Client_Name'])
    return client_name_map

# Save recommendations to file
def save_recommendations_to_file_reverse(products, product_map, recommendations, client_name_map, test, top_k, file_name="recommendations_reverse.txt"):
    with open(file_name, 'w') as f:
        for product in products:
            product_name = product_map[product]
            f.write(f'Product Name: {product_name}\n')
            f.write('Top 10 Recommended Clients:\n')
            product_data = test[test['Product_ID'] == product]
            product_recommendations = product_data.sort_values(by='Normalized_Score', ascending=False)
            unique_recommendations = product_recommendations.drop_duplicates(subset='Client_ID').head(top_k)
            recommendations_list = []
            for _, row in unique_recommendations.iterrows():
                client_id = row['Client_ID']
                client_name = client_name_map[client_name_map['Client_ID'] == client_id]['Client_Name'].values[0]
                recommendations_list.append([client_name, row['Normalized_Score']])
            f.write(tabulate(recommendations_list, headers=['Client Name', 'Predicted Score'], tablefmt='grid'))
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
    recommendations = generate_recommendations_reverse(test, top_k)

    product_map = map_product_ids_to_names(data)
    client_name_map = decode_client_ids_to_names(client_name_map, client_encoder)

    products = test['Product_ID'].unique()
    save_recommendations_to_file_reverse(products, product_map, recommendations, client_name_map, test, top_k)

# Run the main function
file_path = "/Users/igormol/Desktop/analytics/clients/Marajo/sample/Marajo.csv"
main(file_path)
