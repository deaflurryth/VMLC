import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def linear_regression_generator(id):
    np.random.seed(0)
    x = np.random.rand(100, 1)  
    y = 2 * x + 1 + np.random.randn(100, 1) * 0.5 

    df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})

    csv_file = f'generator/generated/linear_regression_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
#linear_regression_generator(5)

def decision_tree_random_forest_generator(id):
    np.random.seed(0)
    x = np.random.rand(100, 2) 
    y = (x[:, 0] + x[:, 1] > 1).astype(int)

    df = pd.DataFrame({'Feature1': x[:, 0], 'Feature2': x[:, 1], 'Target': y})

    csv_file = f'generator/generated/decision_tree_random_forest_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
#decision_tree_random_forest_generator(3)
def svm_dataset_generator(id):
    np.random.seed(0)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Target': y})

    csv_file = f'generator/generated/svm_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
#svm_dataset_generator(1)

def knn_dataset_generator(id):
    np.random.seed(0)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Target': y})

    csv_file = f'generator/generated/knn_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
#knn_dataset_generator(1)

def linear_regression_generator_modern(id, n_features=8):
    np.random.seed(0)
    X, y = make_regression(n_samples=100, n_features=n_features, n_informative=n_features, noise=0.5, random_state=0)

    df = pd.DataFrame({'Feature{}'.format(i): X[:, i] for i in range(n_features)})
    df['Target'] = y

    csv_file = f'generator/generated/linear_regression_data_features{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
linear_regression_generator_modern(1)
def decision_tree_random_forest_generator_modern(id, n_features=5):
    np.random.seed(0)
    X, y = make_classification(n_samples=100, n_features=n_features, n_informative=n_features-1, n_redundant=0, random_state=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X) 

    df = pd.DataFrame({'Feature{}'.format(i): X[:, i] for i in range(n_features)})
    df['Target'] = y

    csv_file = f'generator/generated/decision_tree_random_forest_data_features{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
#decision_tree_random_forest_generator(3)

def logistic_regression_dataset_generator(id):
    np.random.seed(0)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0, n_clusters_per_class=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Target': y})

    csv_file = f'generator/generated/logistic_regression_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
logistic_regression_dataset_generator(1)
def kmeans_dataset_generator(id):
    np.random.seed(0)
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1]})

    csv_file = f'generator/generated/cluster_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
kmeans_dataset_generator(1)
def gradient_boosting_dataset_generator(id):
    np.random.seed(0)
    X, y = make_regression(n_samples=100, n_features=2, n_informative=2, random_state=0, noise=10.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Target': y})

    csv_file = f'generator/generated/gradient_boosting_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
gradient_boosting_dataset_generator(1)

def neural_network_dataset_generator(id):
    np.random.seed(0)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=0)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Target': y})

    csv_file = f'generator/generated/neural_network_data{id}.csv'
    df.to_csv(csv_file, index=False)

    return csv_file
neural_network_dataset_generator(1)