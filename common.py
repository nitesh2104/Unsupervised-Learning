"""
All the common imports, methods and Global variables go here
"""

import numpy as np
import pandas as pd
import time
import mlrose_hiive
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# All the sklearn imports are added below
from sklearn.cluster import KMeans 
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 0 
DECAYS = [mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay, mlrose_hiive.ArithDecay]
output_directory = "outputs"


"""
All the common methods are added down below
"""
# Read dataset for phone price evaluation
def read_dataset(path_to_file=None):
    """
    ## Dataset - # https://www.kaggle.com/iabhishekofficial/mobile-price-classification
    ### Input Features: Set of features (screen size, weight, ram, battery, etc..)
    ### Output Label: Mobile Price (Grouped from 0 to 3)
    """
    df = pd.read_csv(path_to_file)
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    drop_indices = np.random.choice(np.arange(0, df.shape[0], 1), 350, replace=False)
    df.drop(drop_indices, inplace=True)
    X = df.drop(columns=['price_range'])
    y = df[['price_range']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return train_test_split(X, y, test_size=0.2)

# Read dataset for income evaluation
def read_dataset_income(path_to_file=None):
    """
    ## Dataset - # https://www.kaggle.com/iabhishekofficial/mobile-price-classification
    ### Input Features: Set of features (screen size, weight, ram, battery, etc..)
    ### Output Label: Mobile Price (Grouped from 0 to 3)
    """
    df = pd.read_csv(path_to_file)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.columns = list(map(lambda x: x.strip(), df.columns))
    #     df.drop(drop_indices, inplace=True)
    X = df.drop(columns=['income'])
    price_range = df[['income']].to_numpy()
    price_range[price_range=="<=50K"] = 0
    price_range[price_range==">50K"] = 1
    df['income'] = pd.DataFrame(price_range, columns=['income'])
    y = pd.DataFrame(price_range, columns=['income'])
    # Do a one-hot encoding of string features - otherwise the predict method will fail
    for i in df.columns:
        if i in ['workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']:
            df[i] = LabelEncoder().fit_transform(df[i])
    X = df.drop(columns=['income', 'fnlwgt', 'education-num', 'relationship', 'capital-gain', 'capital-loss'])
    y=y.astype('int')
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    return train_test_split(X, y, test_size=0.2)

def plot_graph(x_data, y_data, title, x=None, y=None, color=None):
    """Function to plot the fitness vs iteration values"""
    if not x:
        x="# of Iterations" 
    if not y:
        y="Fitness Val"
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.plot(x_data, y_data, color=color, label=y, lw=2)
    plt.legend(loc="best")
    plt.grid()
    plt.legend()
    plt.show()

def plot_fitness_iteration(file_path, group_by_label=None, x_labels=None, y_labels=None, x=None, y=None, color=None):
    """
    Read the curves
    Group by value groups if there are multiple iterations
    mean() is then used to average out the values
    Data is plotted against the list of y_labels
    """
    df=pd.read_csv(file_path)
    pd.set_option('display.max_rows', df.shape[0])
    df_mean = df.groupby(group_by_label).mean()
    if not x_labels:
        x_labels = range(df_mean.shape[0])
    else:
        x_labels = df_mean[x_labels]
    plot_graph(x_data=x_labels, y_data=df_mean[y_labels], title="Iterations vs Fitness", x=x, y=y, color=color)