import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import random

# Load the CSV file
file_path = "/Users/igormol/Desktop/analytics3/sample/Marajo.csv"
data = pd.read_csv(file_path)

# Fill missing values
data.fillna('', inplace=True)

# Encode categorical features
label_encoders = {}
for column in ['FILIAL', 'Client_ID', 'Product_ID', 'Product_Name', 'Brand_Name']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Create user-item interaction matrix
user_item_matrix = data.pivot_table(index='Client_ID', columns='Product_ID', values='Monetary_Value', fill_value=0)

# Prepare training data
X = []
y = []
for client in user_item_matrix.index:
    for product in user_item_matrix.columns:
        X.append([client, product])
        y.append(user_item_matrix.loc[client, product])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, n_jobs=-1)
model.fit(X_train, y_train)

# Generate recommendations for each client
recommendations = defaultdict(list)
clients = user_item_matrix.index
products = user_item_matrix.columns

for client in clients:
    client_id = label_encoders['Client_ID'].transform([client])[0]
    for product in products:
        product_id = label_encoders['Product_ID'].transform([product])[0]
        score = model.predict(np.array([[client_id, product_id]]))[0]
        recommendations[client].append((product, score))

    # Sort products by score in descending order
    recommendations[client].sort(key=lambda x: x[1], reverse=True)

    # Select top 10 recommendations and add randomness
    top_10_recommendations = recommendations[client][:10]
    random.shuffle(top_10_recommendations)
    recommendations[client] = top_10_recommendations

# Compute recall at 10
def compute_recall_at_10(actual, predicted):
    actual_set = set(actual)
    predicted_set = set(predicted)
    hits = len(actual_set & predicted_set)
    return hits / len(actual_set)

recall_scores = []
for client in clients:
    actual_products = user_item_matrix.loc[client][user_item_matrix.loc[client] > 0].index.tolist()
    predicted_products = [product for product, _ in recommendations[client]]
    recall_scores.append(compute_recall_at_10(actual_products, predicted_products))

average_recall_at_10 = np.mean(recall_scores)

# Prepare the result for saving
result = []
for client in recommendations:
    for product, score in recommendations[client]:
        product_name = label_encoders['Product_Name'].inverse_transform([product])[0]
        result.append([client, product_name, score, average_recall_at_10])

# Save the result as a CSV file
result_df = pd.DataFrame(result, columns=['Client_ID', 'Recommended_Product', 'Predicted_Score', 'Recall_at_10'])
output_file_path = "/Users/igormol/Desktop/analytics3/sample/gboost_recommendations.csv"
result_df.to_csv(output_file_path, index=False)
