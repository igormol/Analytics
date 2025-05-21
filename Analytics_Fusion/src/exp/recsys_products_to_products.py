import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st

def recsys_products_to_products_load_data(file):
    df = pd.read_csv(file)
    df['Sales_Date'] = pd.to_datetime(df['Sales_Date'], errors='coerce')
    df['Monetary_Value'] = pd.to_numeric(df['Monetary_Value'], errors='coerce')
    df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
    df.fillna('', inplace=True)
    return df

def recsys_products_to_products_create_user_item_matrix(df):
    user_item_matrix = df.pivot_table(index='Client_ID', columns='Product_Name', values='Monetary_Value', aggfunc='sum', fill_value=0)
    return user_item_matrix

def recsys_products_to_products_fit_knn_model(matrix):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(matrix)
    return model_knn

def recsys_products_to_products_get_top_n_recommendations(product_name, product_user_matrix, model_knn, n=10):
    if product_name not in product_user_matrix.index:
        return []

    product_vector = product_user_matrix.loc[[product_name]].values
    distances, indices = model_knn.kneighbors(product_vector, n_neighbors=n+1)
    recommended_products = [(product_user_matrix.index[i], distances.flatten()[j]) for j, i in enumerate(indices.flatten()) if product_user_matrix.index[i] != product_name]
    return recommended_products[:n]

def run_recsys_product_to_product(df):
    st.header("Frequently Bought Items")

    product_names = df['Product_Name'].unique()
    selected_product = st.selectbox("Select a product to get recommendations", [""] + list(product_names))

    if selected_product and selected_product != "":
        user_item_matrix = recsys_products_to_products_create_user_item_matrix(df)
        product_user_matrix = user_item_matrix.T
        model_knn = recsys_products_to_products_fit_knn_model(product_user_matrix)

        recommendations = recsys_products_to_products_get_top_n_recommendations(selected_product, product_user_matrix, model_knn)

        if recommendations:
            st.write(f"Recommendations for '{selected_product}':")
            st.table(pd.DataFrame(recommendations, columns=["Recommended Product", "Distance"]))
        else:
            st.write(f"No recommendations found for '{selected_product}'.")

def main():
    st.title("Product Data Analysis")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = recsys_products_to_products_load_data(uploaded_file)
        st.write("Data loaded successfully!")

        with st.sidebar:
            selected_tab = st.radio("Select a tab", ["Home", "Recommendations"])

        if selected_tab == "Recommendations":
            run_recsys_product_to_product(df)

if __name__ == "__main__":
    main()
