# abc_analysis.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from matplotlib.patches import ConnectionPatch

def abc_analysis_aggregate_sales(data):
    # Group the data by 'Product_ID', taking the first occurrence of 'Product_Name' and summing 'Monetary_Value'.
    product_sales = data.groupby('Product_ID').agg({
        'Product_Name': 'first',
        'Monetary_Value': 'sum'
    }).reset_index()
    # Return the aggregated sales data.
    return product_sales

def abc_analysis_calculate_cumulative_sales(product_sales):
    # Sort the product sales by 'Monetary_Value' in descending order.
    product_sales = product_sales.sort_values(by='Monetary_Value', ascending=False)
    # Calculate the total sales.
    total_sales = product_sales['Monetary_Value'].sum()
    # Calculate cumulative sales and cumulative percentage.
    product_sales['Cumulative_Sales'] = product_sales['Monetary_Value'].cumsum()
    product_sales['Cumulative_Percentage'] = (product_sales['Cumulative_Sales'] / total_sales) * 100

    # Assign ABC classes and calculate the contribution metric.
    product_sales['Class'] = product_sales.apply(abc_analysis_assign_abc_class, axis=1)
    product_sales['Metric'] = (product_sales['Monetary_Value'] / total_sales) * 100

    # Return the product sales with cumulative metrics and ABC classes.
    return product_sales

def abc_analysis_assign_abc_class(row):
    # Assign 'A', 'B', or 'C' class based on the cumulative percentage.
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

def abc_analysis_plot_bar_chart(product_sales):
    # Calculate the percentage of products in each ABC class.
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    # Calculate the absolute counts of products in each ABC class.
    absolute_counts = product_sales['Class'].value_counts()
    # Ensure order A, B, C for consistency in plotting.
    class_counts.sort_index(inplace=True)
    # Calculate the mean metric for each ABC class.
    mean_metric = product_sales.groupby('Class')['Metric'].mean()

    # Define colors for the bars representing each ABC class.
    colors = ['#E07A5F', '#3D405B', '#81B29A']

    # Create a bar chart with specified figure size.
    plt.figure(figsize=(10, 6))
    # Plot the bars representing each ABC class.
    bars = plt.bar(class_counts.index, class_counts.values, color=colors)
    # Set labels for y-axis and title of the bar chart.
    plt.ylabel('Percentage of Products')
    plt.title('ABC Classification of Products')

    # Annotate each bar with its corresponding percentage, absolute count, and mean metric.
    for bar, percentage, metric, absolute in zip(bars, class_counts.values, mean_metric, absolute_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f'{percentage:.2f}%\n{absolute} products\nMean Metric: {metric:.2f}',
                 ha='center', color='black')

    # Set y-axis limits to ensure the range from 0 to 100.
    plt.ylim(0, 100)

    # Save the bar chart as a PNG image.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.pyplot(plt)
    plt.close()

    # Return the binary data of the saved image.
    return buf.getvalue()

def abc_analysis_plot_pie_chart(product_sales):
    # Calculate the percentage of products in each ABC class.
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    # Calculate the absolute counts of products in each ABC class.
    absolute_counts = product_sales['Class'].value_counts()
    # Calculate the mean metric for each ABC class.
    mean_metric = product_sales.groupby('Class')['Metric'].mean()
    # Ensure order A, B, C for consistency in plotting.
    class_counts.sort_index(inplace=True)

    # Define colors for the wedges representing each ABC class.
    colors = ['#E07A5F', '#3D405B', '#81B29A']

    # Create a pie chart with specified figure size.
    plt.figure(figsize=(8, 8))
    # Plot the pie chart with wedges representing each ABC class, showing percentages and using defined colors.
    wedges, texts, autotexts = plt.pie(class_counts, labels=None, autopct='%1.1f%%', startangle=140, colors=colors)

    # Add labels outside the pie chart for each wedge.
    for i, (wedge, percentage, absolute, metric) in enumerate(zip(wedges, class_counts.values, absolute_counts.values, mean_metric)):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        plt.annotate(f'{class_counts.index[i]}\n{percentage:.1f}%\n{absolute} products\nMean Metric: {metric:.2f}',
                     xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                     horizontalalignment=horizontalalignment,
                     bbox=dict(boxstyle="square,pad=0.3", edgecolor="black", facecolor="white"),
                     arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle))

    # Set the title of the pie chart.
    plt.title('ABC Classification of Products')
    plt.tight_layout()  # Adjust layout to fit labels

    # Save the pie chart as a PNG image.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.pyplot(plt)
    plt.close()

    # Return the binary data of the saved image.
    return buf.getvalue()
