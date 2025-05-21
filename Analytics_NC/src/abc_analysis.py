# abc_analysis.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from matplotlib.patches import ConnectionPatch

# Function to aggregate sales data by Product_ID
def abc_analysis_aggregate_sales(data):
    # Group data by Product_ID and aggregate product name and monetary value
    product_sales = data.groupby('Product_ID').agg({
        'Product_Name': 'first',  # Take the first product name for each product ID
        'Monetary_Value': 'sum'  # Sum the monetary values for each product ID
    }).reset_index()  # Reset index to get a clean DataFrame
    return product_sales  # Return the aggregated DataFrame

# Function to calculate cumulative sales and assign ABC classes
def abc_analysis_calculate_cumulative_sales(product_sales):
    # Sort products by monetary value in descending order
    product_sales = product_sales.sort_values(by='Monetary_Value', ascending=False)
    # Calculate the total sales
    total_sales = product_sales['Monetary_Value'].sum()
    # Calculate cumulative sales
    product_sales['Cumulative_Sales'] = product_sales['Monetary_Value'].cumsum()
    # Calculate cumulative percentage of total sales
    product_sales['Cumulative_Percentage'] = (product_sales['Cumulative_Sales'] / total_sales) * 100
    # Assign ABC classes based on cumulative percentage
    product_sales['Class'] = product_sales.apply(abc_analysis_assign_abc_class, axis=1)
    # Calculate the percentage of total sales for each product
    product_sales['Metric'] = (product_sales['Monetary_Value'] / total_sales) * 100
    return product_sales  # Return the updated DataFrame

# Function to assign ABC class based on cumulative percentage
def abc_analysis_assign_abc_class(row):
    # Class A if cumulative percentage is <= 70%
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    # Class B if cumulative percentage is > 70% and <= 90%
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    # Class C if cumulative percentage is > 90%
    else:
        return 'C'

# Function to plot and save a bar chart of the ABC classification
def abc_analysis_plot_bar_chart(product_sales):
    # Calculate the percentage of products in each class
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    # Count the absolute number of products in each class
    absolute_counts = product_sales['Class'].value_counts()
    # Ensure order A, B, C
    class_counts.sort_index(inplace=True)
    # Calculate the mean metric for each class
    mean_metric = product_sales.groupby('Class')['Metric'].mean()

    # Define color scheme
    colors = ['#E07A5F', '#3D405B', '#81B29A']
    # Set the figure size
    plt.figure(figsize=(10, 6))
    # Plot bar chart
    bars = plt.bar(class_counts.index, class_counts.values, color=colors)
    # Set y-axis label
    plt.ylabel('Percentage of Products')
    # Set chart title
    plt.title('ABC Classification of Products')

    # Annotate each bar with percentage, count, and mean metric
    for bar, percentage, metric, absolute in zip(bars, class_counts.values, mean_metric, absolute_counts):
        # Get the height of each bar
        height = bar.get_height()
        # Add text annotation
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f'{percentage:.2f}%\n{absolute} products\nMean Metric: {metric:.2f}',
                 ha='center', color='black')

    # Set y-axis range to 0-100%
    plt.ylim(0, 100)
    # Create a buffer to save the plot
    buf = io.BytesIO()
    # Save the plot to the buffer
    plt.savefig(buf, format='png')
    # Rewind the buffer
    buf.seek(0)
    # Display the plot in Streamlit
    st.pyplot(plt)
    # Close the plot to free memory
    plt.close()
    # Return the plot as a PNG image
    return buf.getvalue()

# Function to plot and save a pie chart of the ABC classification with labels outside the pie
def abc_analysis_plot_pie_chart(product_sales):
    # Calculate the percentage of products in each class
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    # Count the absolute number of products in each class
    absolute_counts = product_sales['Class'].value_counts()
    # Calculate the mean metric for each class
    mean_metric = product_sales.groupby('Class')['Metric'].mean()
    # Ensure order A, B, C
    class_counts.sort_index(inplace=True)

    # Define color scheme
    colors = ['#E07A5F', '#3D405B', '#81B29A']
    # Set the figure size
    plt.figure(figsize=(8, 8))
    # Plot pie chart
    wedges, texts, autotexts = plt.pie(class_counts, labels=None, autopct='%1.1f%%', startangle=140, colors=colors)

    # Add labels outside the pie
    for i, (wedge, percentage, absolute, metric) in enumerate(zip(wedges, class_counts.values, absolute_counts.values, mean_metric)):
        # Calculate angle for each wedge
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        # Calculate y-coordinate for label
        y = np.sin(np.deg2rad(ang))
        # Calculate x-coordinate for label
        x = np.cos(np.deg2rad(ang))
        # Set horizontal alignment based on x-coordinate
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        # Define connection style for annotation
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        # Annotate each wedge with percentage, count, and mean metric
        plt.annotate(f'{class_counts.index[i]}\n{percentage:.1f}%\n{absolute} products\nMean Metric: {metric:.2f}',
                     xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                     horizontalalignment=horizontalalignment,
                     bbox=dict(boxstyle="square,pad=0.3", edgecolor="black", facecolor="white"),
                     arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle))

    # Set chart title
    plt.title('ABC Classification of Products')
    # Adjust layout to fit labels
    plt.tight_layout()
    # Create a buffer to save the plot
    buf = io.BytesIO()
    # Save the plot to the buffer
    plt.savefig(buf, format='png')
    # Rewind the buffer
    buf.seek(0)
    # Display the plot in Streamlit
    st.pyplot(plt)
    # Close the plot to free memory
    plt.close()
    # Return the plot as a PNG image
    return buf.getvalue()
