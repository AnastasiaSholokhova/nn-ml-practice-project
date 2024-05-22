"""Modules"""
import os
import pandas as pd

def data_load(
        x_train_path : str, y_train_path : str, x_test_path : str,
        y_test_path : str) -> pd.DataFrame:
    """
    Loading data
    :param x_train_path: 
    :type x_train_path: str
    :param y_train_path: 
    :type y_train_path: str
    :param x_test_path: 
    :type x_test_path: str
    :param y_test_path: 
    :type y_test_path: str
    :rtype: pd.DataFrame

    """
    x_train = pd.read_csv(x_train_path, delimiter=',')
    y_train = pd.read_csv(y_train_path, delimiter=',')
    x_test = pd.read_csv(x_test_path, delimiter=',')
    y_test = pd.read_csv(y_test_path, delimiter=',')
    return x_train, y_train, x_test, y_test

project_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_dir = os.path.join(project_dir, "datasets\\csv\\preprocessed\\")
os.makedirs(preprocessed_dir, exist_ok=True)

x_train_path = os.path.join(project_dir, "datasets\\csv\\input.csv")
y_train_path = os.path.join(project_dir, "datasets\\csv\\labels.csv")
x_test_path = os.path.join(project_dir, "datasets\\csv\\input_test.csv")
y_test_path = os.path.join(project_dir, "datasets\\csv\\labels_test.csv")

x_train, y_train, x_test, y_test = data_load(x_train_path, y_train_path, x_test_path, y_test_path)

x_train.to_csv(os.path.join(preprocessed_dir, "preprocessed_input.csv"), index=False)
y_train.to_csv(os.path.join(preprocessed_dir, "preprocessed_labels.csv"), index=False)
x_test.to_csv(os.path.join(preprocessed_dir, "preprocessed_input_test.csv"), index=False)
y_test.to_csv(os.path.join(preprocessed_dir, "preprocessed_labels_test.csv"), index=False)
