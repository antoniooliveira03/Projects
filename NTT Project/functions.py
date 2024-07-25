import matplotlib.pyplot as plt
import numpy as np

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