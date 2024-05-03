import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori
import plotly.express as px

def graphs(data, graph, color='orange'):
    """
    Generate boxplots or histograms for numeric columns in the provided data.

    Args:
      data (pandas.DataFrame): The input data containing numeric columns.
      graph (str): The type of graph to generate. Options: 'boxplot' or 'histogram'.
      color (str): The color of the plot. Default is 'orange'.
    """

    for column in data.columns:
        # Check if the column is numeric
        if data[column].dtype == float or data[column].dtype == int:
            if graph == 'boxplot':
                plt.boxplot(data[column], vert=False, patch_artist=True, boxprops=dict(facecolor=color))
                plt.title(f'Boxplot of {column}')
                plt.yticks([])
                plt.show()

            elif graph == 'histogram':
                # Check if there are no infinite values in the column
                if not np.isinf(data[column]).any():
                    plt.hist(data[column], color=color)
                    plt.title(f'Histogram of {column}')
                    plt.show()


def calculate_age(birth_date): 
    """
    Calculates the age based on the birth date provided
    """
    current_date = datetime.datetime.now()
    age = current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
    return age


def cor_heatmap(cor):
    """
    Generate a heatmap to visualize the correlation matrix.

    Parameters:
        cor (pandas.DataFrame): The correlation matrix to be visualized.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(data=cor, annot=True, cmap=plt.cm.Oranges, fmt='.1')
    plt.show()


 
def plot_histogram(data, xlabel, ylabel, title, color='orange'):
    """
    Plots a histogram of the given data.

    Parameters:
        data (array-like): The data to be plotted.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the histogram.
        color (str): The color of the histogram bars. Default is 'blue'.
    """
    plt.hist(data, edgecolor='black', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_pie_chart(data, column, colors=None, subtitles=None):
    """
    Generate a pie chart to visualize the distribution of a categorical variable.

    Parameters:
        data (pandas.DataFrame): The dataset containing the column to be visualized.
        column (str): The column name representing the categorical variable.
        colors (list, optional): A list of colors for the pie chart slices. Defaults to None.
        subtitles (list, optional): A list of subtitles corresponding to each category. Defaults to None.
    """
    value_counts = data[column].value_counts()
    labels = value_counts.index.tolist()
    sizes = value_counts.values.tolist()

    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    ax.axis('equal')
    plt.title(f'Pie Chart of {column}')

    if subtitles:
        legend_labels = [f'{label} - {subtitle}' for label, subtitle in zip(labels, subtitles)]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def visualize_dimensionality_reduction(transformation, targets):
    """
       Visualizes the dimensionality reduction results using a scatter plot.
  
       Args:
           transformation (np.ndarray): The transformed data points.
           targets (List[Any]): The target labels or categories for each data point.
  
       Returns:
           None
       """
    # create a scatter plot of the UMap output
    plt.scatter(transformation[:, 0], transformation[:, 1], 
              color=plt.cm.tab10(np.array(targets).astype(int)))

    labels = np.unique(targets)

    # create a legend with the class labels and colors
    handles = [plt.scatter([],[], color=plt.cm.tab10(i), label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Clusters')

    plt.show()


def product_rules(cluster):
    """
    Extracts frequent itemsets and association rules from a cluster of products.

    Parameters:
        cluster (DataFrame): A DataFrame representing a cluster of products.
            It should contain the following columns:
            - 'list_of_goods': A column containing lists of products in each transaction.

    Returns:
        None, but 2 DataFrames containing the top 10 association rules based on lift value and
        top 10 products are shown.

    """
    list_products = [ast.literal_eval(product) for product in cluster['list_of_goods'].values]

    te = TransactionEncoder()
    te_fit = te.fit(list_products).transform(list_products)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)

    frequent_itemsets_grocery = apriori(
    transactions_items, min_support=0.05, use_colnames=True
    )
  
    display(frequent_itemsets_grocery.sort_values(by='support', ascending=False).head(10))

    frequent_itemsets_grocery_iter = apriori(
    transactions_items, min_support=0.02, use_colnames=True
    )

    # We'll use a confidence level of 20%
    rules_grocery_iter = association_rules(frequent_itemsets_grocery_iter, 
                                  metric="confidence", 
                                  min_threshold=0.2)
  
    display(rules_grocery_iter.sort_values(by='lift', ascending=False).head(10))


def category_rules(cluster,categories):
    
    """
    Extracts frequent itemsets and association rules from a cluster of products.

    Parameters:
        cluster (DataFrame): A DataFrame representing a cluster of products.
            It should contain the following columns:
            - 'list_of_goods': A column containing lists of products in each transaction.
        categories (DataFrame): A DataFrame representing the categories of each product.
            It should contain the following columns:
            - 'product_name': A column containing the names of each product.
            - 'category': A column containing the categories of each product.

    Returns:
       None, but The function displays the frequent itemsets and association rules.

    """

    list_products = [ast.literal_eval(product) for product in cluster['list_of_goods'].values]
    list_categories = [[categories.loc[categories['product_name'] == item, 'category'].values[0] for item in sublist] for sublist in list_products]
    te = TransactionEncoder()
    te_fit = te.fit(list_categories).transform(list_categories)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)

    frequent_itemsets_grocery = apriori(
    transactions_items, min_support=0.05, use_colnames=True
    )
  
    display(frequent_itemsets_grocery.sort_values(by='support', ascending=False).head(10))

    frequent_itemsets_grocery_iter = apriori(
    transactions_items, min_support=0.02, use_colnames=True
    )

    # We'll use a confidence level of 20%
    rules_grocery_iter = association_rules(frequent_itemsets_grocery_iter, 
                                  metric="confidence", 
                                  min_threshold=0.2)
  
    display(rules_grocery_iter.sort_values(by='lift', ascending=False).head(10))


def map_cluster(cluster):
    """
      Plots a scatter plot on a map for a given cluster.

      Args:
          cluster (DataFrame): DataFrame containing latitude and longitude coordinates.

      Returns:
          None, but each observation in the cluster is shown in a map
      """
    # Plot the map
    fig = px.scatter_mapbox(lat=cluster['latitude'],
                            lon=cluster['longitude'],
                            zoom=10)

    fig.update_traces(marker=dict(size=4, color='orange'))

    # Set the map style
    fig.update_layout(mapbox_style="stamen-terrain")

    # Show the map
    fig.show() 


