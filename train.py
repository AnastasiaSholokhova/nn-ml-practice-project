"""Modules providing model training"""
import pickle
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

import pickle

def train_model(x_train_preprocessed: pd.DataFrame, y_train_preprocessed: pd.DataFrame) -> pickle:
    """
    Model training
    :param x_train_preprocessed: 
    :type x_train_preprocessed: pd.DataFrame
    :param y_train_preprocessed: 
    :type y_train_preprocessed: pd.DataFrame
    :rtype: pickle

    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_preprocessed, y_train_preprocessed, epochs=5, batch_size=64)

    model_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(model_path, "models")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file_path = os.path.join(model_dir, "model.pkl")
    model.save(model_file_path)
    return model

x_train_preprocessed = pd.read_csv("datasets\\csv\\preprocessed\\preprocessed2_input.csv")
x_train_preprocessed = x_train_preprocessed.values.reshape(len(x_train_preprocessed), 100, 100, 3)
y_train_preprocessed = pd.read_csv("datasets\\csv\\preprocessed\\preprocessed2_labels.csv")
model = train_model(x_train_preprocessed, y_train_preprocessed)
