import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def hybrid_recsys_load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def hybrid_recsys_preprocess_data(data):
    data['Brand_Name'] = data['Brand_Name'].fillna('')
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    data['user_id'] = user_encoder.fit_transform(data['Client_ID'])
    data['item_id'] = item_encoder.fit_transform(data['Product_ID'])
    return data, user_encoder, item_encoder

def hybrid_recsys_aggregate_data(data):
    data_aggregated = data.groupby(['user_id', 'item_id'], as_index=False)['Monetary_Value'].sum()
    return data_aggregated

def hybrid_recsys_create_interaction_matrix(data_aggregated):
    interaction_matrix = data_aggregated.pivot(index='user_id', columns='item_id', values='Monetary_Value').fillna(0)
    return interaction_matrix

def hybrid_recsys_train_test_split(interaction_matrix):
    train_data, test_data = train_test_split(interaction_matrix, test_size=0.2, random_state=42)
    return train_data, test_data

def hybrid_recsys_train_autoencoder(train_data, test_data, input_dim, encoding_dim=10):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(train_data, train_data, epochs=10, batch_size=128, shuffle=True, validation_data=(test_data, test_data), verbose=1)
    encoder = Model(input_layer, encoded)
    encoded_train_data = encoder.predict(train_data)
    return encoded_train_data, encoder

def hybrid_recsys_train_xgboost(encoded_train_data, train_data):
    x_train = encoded_train_data
    y_train = train_data.values
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, max_depth=5, learning_rate=0.1)
    model.fit(x_train, y_train)
    return model

def hybrid_recsys_generate_recommendations(user_name, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix, data):
    user_id = user_encoder.transform([user_name])[0]
    user_vector = encoded_train_data[user_id].reshape(1, -1)
    predicted_ratings = model.predict(user_vector).flatten()
    assert len(predicted_ratings) == interaction_matrix.shape[1], "Predicted ratings length mismatch."
    recommendations = pd.DataFrame({'item_id': range(len(predicted_ratings)), 'predicted_rating': predicted_ratings})
    top_10_recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(10)
    top_10_recommendations['Product_ID'] = item_encoder.inverse_transform(top_10_recommendations['item_id'])
    top_10_recommendations = top_10_recommendations.merge(data[['Product_ID', 'Product_Name']].drop_duplicates(), on='Product_ID')
    return top_10_recommendations

def main():
    file_path = "/Users/igormol/Desktop/analytics3/sample/Marajo.csv"
    data = hybrid_recsys_load_data(file_path)
    data, user_encoder, item_encoder = hybrid_recsys_preprocess_data(data)
    data_aggregated = hybrid_recsys_aggregate_data(data)
    interaction_matrix = hybrid_recsys_create_interaction_matrix(data_aggregated)
    train_data, test_data = hybrid_recsys_train_test_split(interaction_matrix)
    input_dim = interaction_matrix.shape[1]
    encoded_train_data, encoder = hybrid_recsys_train_autoencoder(train_data, test_data, input_dim)
    model = hybrid_recsys_train_xgboost(encoded_train_data, train_data)
    user_name = "ADMILSON VITOR DE SOUZA"
    top_10_recommendations = hybrid_recsys_generate_recommendations(user_name, user_encoder, encoded_train_data, model, item_encoder, interaction_matrix, data)
    print(top_10_recommendations[['Product_ID', 'Product_Name', 'predicted_rating']])

if __name__ == "__main__":
    main()
