# main.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import streamlit as st
from webinterface import *

def main():
    # Set the title of the Streamlit app
    st.title('Makes Analytics NC (Neural Collab)')

    # Create a file uploader widget for uploading CSV files
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load data from the uploaded CSV file using a function called "rfm_clients_load_data"
        data = rfm_clients_load_data(uploaded_file)

        # Set the title of the sidebar
        st.sidebar.title("Navigation")

        # Create a radio button widget for selecting different app modes
        app_mode = st.sidebar.radio("Choose the app mode", [
            "ABC Analysis",  # Option for ABC Analysis
            "RFM Analysis for Clients",  # Option for RFM Analysis for Clients
            "RFM Analysis for Products",  # Option for RFM Analysis for Products
            "Client Retention Analysis",  # Option for Client Retention Analysis
            "Recommendations of Clients to Products",  # Option for Recommendations of Clients to Products
            "Recommendations of Products to Clients",  # Option for Recommendations of Products to Clients
            "Frequently Bought Together"               # Option for Recommendations of Products to Products
        ])

        # Check the selected app mode and execute the corresponding function
        if app_mode == "RFM Analysis for Clients":
            run_rfm_clients(data)
        elif app_mode == "RFM Analysis for Products":
            run_rfm_products(data)
        elif app_mode == "ABC Analysis":
            run_abc_analysis(data)
        elif app_mode == "Client Retention Analysis":
            run_churn_analysis(data)
        elif app_mode == "Recommendations of Clients to Products":
            run_recsys_clients_to_products(data)
        elif app_mode == "Recommendations of Products to Clients":
            run_recsys_products_to_clients(data)
        elif app_mode == "Frequently Bought Together":
            run_recsys_product_to_product(data)

if __name__ == "__main__":
    main()
