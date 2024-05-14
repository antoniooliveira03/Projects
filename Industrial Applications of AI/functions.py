import matplotlib.pyplot as plt
import seaborn as sns

from math import radians, sin, cos, sqrt, atan2

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# boxplots

def plot_boxplots(data, columns_to_check, palette=None):
    for column in columns_to_check:
        plt.figure(figsize=(6, 4))  
        sns.boxplot(data=data[column], palette=palette)
        plt.title(f'Box Plot of {column}')
        plt.xticks()
        plt.show()

# distance between 2 points

def haversine(lat1, lon1, lat2, lon2):
    
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Calculate the distance
    distance = R * c
    
    return distance

def compute_distance(row):
    lat1 = row['Restaurant_latitude']
    lon1 = row['Restaurant_longitude']
    lat2 = row['Delivery_location_latitude']
    lon2 = row['Delivery_location_longitude']
    return haversine(lat1, lon1, lat2, lon2)


# model evaluation
def evaluator(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
