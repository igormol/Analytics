import streamlit as st
from webinterface import *
from recsys_clients import *

def main():
    st.title('RFM Analysis Web App')

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        recsys_clients_data, recsys_clients_client_name_map, recsys_clients_client_encoder, recsys_clients_product_encoder = recsys_clients_load_and_preprocess_data(uploaded_file)
        data = rfm_clients_load_data(uploaded_file)
        product_data, product_client_name_map, product_client_encoder, product_product_encoder = recsys_products_load_and_preprocess_data(uploaded_file)

        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio("Choose the app mode", [
            "ABC Analysis",
            "RFM Analysis for Clients",
            "RFM Analysis for Products",
            "Client Retention Analysis",
            "Recommendations of Clients to Products",
            "Recommendations of Products to Clients"
        ])

        if app_mode == "RFM Analysis for Clients":
            run_rfm_clients(data)
        elif app_mode == "RFM Analysis for Products":
            run_rfm_products(data)
        elif app_mode == "ABC Analysis":
            run_abc_analysis(data)
        elif app_mode == "Client Retention Analysis":
            run_churn_analysis(data)
        elif app_mode == "Recommendations of Clients to Products":
            recsys_clients_recommendations(recsys_clients_data, recsys_clients_client_name_map, recsys_clients_client_encoder, recsys_clients_product_encoder)
        elif app_mode == "Recommendations of Products to Clients":
            recsys_products_main_functionality(product_data, product_client_name_map, product_client_encoder, product_product_encoder)

if __name__ == "__main__":
    main()
