import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import json
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

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


def serialize_list(lst):
    """ serealize list to keep data types after export"""
    return json.dumps(lst)


def deserialize_list(json_str):
    """ deserealize list to keep data types after import"""
    return json.loads(json_str)


def find_non_numeric_values(lst):
    non_numeric_values = [item for item in lst if not isinstance(item, (int, float))]
    return non_numeric_values


def unique_words(words_list):
    return list(set(words_list))


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
    list_products = cluster['product_id'].tolist()

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


def category_rules(cluster,prod):
    
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

    #list_products = [ast.literal_eval(product) for product in cluster['product_id'].values]
    list_products = cluster['product_id'].tolist()
    #list_categories = [[categories.loc[categories['product_id'] == item, 'department'].values[0] for item in sublist] for sublist in list_products]
    list_categories = []
    for sublist in list_products:
        category_list = []
        for item in sublist:
            # Find category for the product ID
            category = prod.loc[prod['product_id'] == item, 'department']
            if not category.empty:
                category_list.append(category.values[0])
            # If no category is found, skip the item
        # Only add non-empty category lists
        if category_list:
            list_categories.append(category_list)

    
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
