"""Libraries"""
import pandas as pd
import keras
import os
import json

def predict(model, x_test_preprocessed, idx):
    """
    Model predictions
    :param model: 
    :param x_test_preprocessed: 
    :param idx: 

    """
    y_pred = model.predict(x_test_preprocessed[idx].reshape(1, 100, 100, 3))
    y_pred = y_pred > 0.5
    pred = 'dog' if y_pred == 0 else 'cat'
    print('Our model says it is a:', pred)

def evaluate_model(model, x_test_preprocessed, y_test_preprocessed):
    """
    Model evaluation
    :param model: 
    :param x_test_preprocessed: 
    :param y_test_preprocessed: 

    """
    loss, accuracy = model.evaluate(x_test_preprocessed, y_test_preprocessed)
    evaluation_results = {
        "loss": loss,
        "accuracy": accuracy
    }
    metrics_path = "metrics" 

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    
    metrics_file_path = os.path.join(metrics_path, "metrics.json")
    json_metrics = json.dumps(evaluation_results)
    
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write(json_metrics)
        f.flush()
    return evaluation_results 

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "models\\model.pkl")
model = keras.models.load_model(model_path)
x_test_preprocessed = pd.read_csv("datasets/csv/preprocessed/preprocessed2_input_test.csv")
x_test_preprocessed = x_test_preprocessed.values.reshape(len(x_test_preprocessed), 100, 100, 3)
y_test_preprocessed = pd.read_csv("datasets/csv/preprocessed/preprocessed2_labels_test.csv")
metrics = evaluate_model(model, x_test_preprocessed, y_test_preprocessed)

idx = int(input("Enter the index of the image you want to predict (0 to {}): ".format(len(x_test_preprocessed)-1)))
pred = predict(model, x_test_preprocessed, idx)
print(pred, metrics)
