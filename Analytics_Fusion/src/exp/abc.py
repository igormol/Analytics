import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(file_path)

def aggregate_sales(data):
    """Aggregate sales data by Product_ID."""
    product_sales = data.groupby('Product_ID').agg({
        'Product_Name': 'first',
        'Monetary_Value': 'sum'
    }).reset_index()
    return product_sales

def calculate_cumulative_sales(product_sales):
    """Calculate cumulative sales and assign ABC classes."""
    product_sales = product_sales.sort_values(by='Monetary_Value', ascending=False)
    total_sales = product_sales['Monetary_Value'].sum()
    product_sales['Cumulative_Sales'] = product_sales['Monetary_Value'].cumsum()
    product_sales['Cumulative_Percentage'] = (product_sales['Cumulative_Sales'] / total_sales) * 100

    product_sales['Class'] = product_sales.apply(assign_abc_class, axis=1)
    product_sales['Metric'] = (product_sales['Monetary_Value'] / total_sales) * 100

    # Calculate confidence index
    product_sales['Confidence_Index'] = (product_sales['Cumulative_Percentage'] - 70) / 2.0
    product_sales['Confidence_Index'] = product_sales['Confidence_Index'].apply(lambda x: max(0, min(100, x)))  # Normalize to 0-100

    return product_sales

def assign_abc_class(row):
    """Assign ABC class based on cumulative percentage."""
    if row['Cumulative_Percentage'] <= 70:
        return 'A'
    elif row['Cumulative_Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

def save_table_to_txt(product_sales, output_file):
    """Save the product classification table to a text file using tabulate."""
    table = tabulate(product_sales[['Product_Name', 'Class', 'Metric', 'Confidence_Index']], headers='keys', tablefmt='grid', showindex=False)
    with open(output_file, 'w') as file:
        file.write(table)

def plot_bar_chart(product_sales, output_file):
    """Plot and save a bar chart of the ABC classification."""
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    class_counts.sort_index(inplace=True)  # Ensure order A, B, C
    mean_metric = product_sales.groupby('Class')['Metric'].mean()
    confidence_index = product_sales.groupby('Class')['Confidence_Index'].mean()

    colors = ['#E07A5F', '#3D405B', '#81B29A']
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.index, class_counts.values, color=colors)
    plt.ylabel('Percentage of Products')
    plt.title('ABC Classification of Products')

    for i, (percentage, metric, confidence) in enumerate(zip(class_counts.values, mean_metric, confidence_index)):
        plt.text(i, percentage + 1, f'Mean Metric: {metric:.2f}\nConfidence Index: {confidence:.2f}', ha='center', color='black')

    plt.ylim(0, 100)
    plt.savefig(output_file)
    plt.close()

def plot_pie_chart(product_sales, output_file):
    """Plot and save a pie chart of the ABC classification."""
    class_counts = product_sales['Class'].value_counts(normalize=True) * 100
    class_counts.sort_index(inplace=True)  # Ensure order A, B, C

    colors = ['#E07A5F', '#3D405B', '#81B29A']
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('ABC Classification of Products')
    plt.savefig(output_file)
    plt.close()

def main():
    file_path = "/Users/igormol/Desktop/analytics/clients/Marajo/sample/Marajo.csv"
    txt_output_file = "/Users/igormol/Desktop/analytics/clients/Marajo/sample/ABC_Classification.txt"
    bar_output_file = "/Users/igormol/Desktop/analytics/clients/Marajo/sample/ABC_Classification_bar.png"
    pie_output_file = "/Users/igormol/Desktop/analytics/clients/Marajo/sample/ABC_Classification_pie.png"

    data = load_data(file_path)
    product_sales = aggregate_sales(data)
    product_sales = calculate_cumulative_sales(product_sales)

    save_table_to_txt(product_sales, txt_output_file)
    plot_bar_chart(product_sales, bar_output_file)
    plot_pie_chart(product_sales, pie_output_file)

if __name__ == "__main__":
    main()
