"""Module providing data preprocessing"""
import pandas as pd
import os

def preprocess(x_train : pd.DataFrame , y_train : pd.DataFrame, x_test : pd.DataFrame,
               y_test : pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing data
    :param X_train: 
    :type X_train: pd.DataFrame
    :param y_train: 
    :type y_train: pd.DataFrame
    :param X_test: 
    :type X_test: pd.DataFrame
    :param y_test: 
    :type y_test: pd.DataFrame
    :rtype: pd.DataFrame

    """
    x_train_processed = x_train.values.reshape(len(x_train), 100, 100, 3)
    y_train_processed = y_train.values.reshape(len(y_train), 1)
    x_test_processed = x_test.values.reshape(len(x_test), 100, 100, 3)
    y_test_processed = y_test.values.reshape(len(y_test), 1)

    x_train_processed = x_train_processed/255.0
    x_test_processed = x_test_processed/255.0
    return x_train_processed, y_train_processed, x_test_processed, y_test_processed


project_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_dir = os.path.join(project_dir, "datasets\\csv\\preprocessed\\")
os.makedirs(preprocessed_dir, exist_ok=True)

x_train = pd.read_csv("datasets/csv/preprocessed/preprocessed_input.csv")
y_train = pd.read_csv("datasets/csv/preprocessed/preprocessed_labels.csv")
x_test = pd.read_csv("datasets/csv/preprocessed/preprocessed_input_test.csv")
y_test = pd.read_csv("datasets/csv/preprocessed/preprocessed_labels_test.csv")

x_train_preprocessed, y_train_preprocessed, x_test_preprocessed, y_test_preprocessed = preprocess(x_train, y_train, x_test, y_test)

x_train_preprocessed_df = pd.DataFrame(x_train_preprocessed)
y_train_preprocessed_df = pd.DataFrame(y_train_preprocessed)
x_test_preprocessed_df = pd.DataFrame(x_test_preprocessed)
y_test_preprocessed_df = pd.DataFrame(y_test_preprocessed)

x_train_preprocessed_df.to_csv(os.path.join(preprocessed_dir, "preprocessed2_input.csv"), index=False)
y_train_preprocessed_df.to_csv(os.path.join(preprocessed_dir, "preprocessed2_labels.csv"), index=False)
x_test_preprocessed_df.to_csv(os.path.join(preprocessed_dir, "preprocessed2_input_test.csv"), index=False)
y_test_preprocessed_df.to_csv(os.path.join(preprocessed_dir, "preprocessed2_labels_test.csv"), index=False)
