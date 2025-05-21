import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from matplotlib.patches import ConnectionPatch

def abc_analysis_aggregate_sales(data):
    """Aggregate sales data by Product_ID."""
    product_sales = data.groupby('Product_ID').agg({
        'Product_Name': 'first',
        'Monetary_Value': 'sum'
    }).reset_index()
    return product_sales

def abc_analysis_calculate_cumulative_sales(product_sales):
    """Calculate cumulative sales and assign ABC classes."""
    product_sales = product_sales.sort_values(by='Monetary_Value', ascending=False)
    total_sales = product_sales['Monetary_Value'].sum()
    product_sales['Cumulative_Sales'] = product_sales['Monetary_Value'].cumsum()
    product_sales['Cumulative_Percentage'] = (product_sales['Cumulative_Sales'] / total_sales) * 100

    product_sales['Class'] = product_sales.apply(abc_analysis_assign_abc_class, axis=1)
    product_sales['Metric'] = (product_sales['Monetary_Value'] / total_sales) * 100

    return product_sales

def abc_analysis_assign_abc_class(row):
    """Assign ABC class based on cumulative percentage."""
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

def abc_analysis_plot_bar_chart(product_sales):
    """Plot and save a bar chart of the ABC classification."""
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    absolute_counts = product_sales['Class'].value_counts()
    class_counts.sort_index(inplace=True)  # Ensure order A, B, C
    mean_metric = product_sales.groupby('Class')['Metric'].mean()

    colors = ['#E07A5F', '#3D405B', '#81B29A']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.index, class_counts.values, color=colors)
    plt.ylabel('Percentage of Products')
    plt.title('ABC Classification of Products')

    for bar, percentage, metric, absolute in zip(bars, class_counts.values, mean_metric, absolute_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f'{percentage:.2f}%\n{absolute} products\nMean Metric: {metric:.2f}',
                 ha='center', color='black')

    plt.ylim(0, 100)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.pyplot(plt)
    plt.close()
    return buf.getvalue()

def abc_analysis_plot_pie_chart(product_sales):
    """Plot and save a pie chart of the ABC classification with labels outside the pie."""
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    absolute_counts = product_sales['Class'].value_counts()
    mean_metric = product_sales.groupby('Class')['Metric'].mean()
    class_counts.sort_index(inplace=True)  # Ensure order A, B, C

    colors = ['#E07A5F', '#3D405B', '#81B29A']
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(class_counts, labels=None, autopct='%1.1f%%', startangle=140, colors=colors)

    # Add labels outside the pie
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

    plt.title('ABC Classification of Products')
    plt.tight_layout()  # Adjust layout to fit labels
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.pyplot(plt)
    plt.close()
    return buf.getvalue()
