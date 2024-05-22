from flask import Flask, jsonify, request
import pandas as pd
import os
import keras

app = Flask("Binary classification")

x_test_preprocessed = pd.read_csv("datasets/csv/preprocessed/preprocessed2_input_test.csv")
x_test_preprocessed = x_test_preprocessed.values.reshape(len(x_test_preprocessed), 100, 100, 3)

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "models/model.pkl")
model = keras.models.load_model(model_path)

def predict(model, x_test_preprocessed, idx):
    y_pred = model.predict(x_test_preprocessed[idx].reshape(1, 100, 100, 3))
    y_pred = y_pred > 0.5
    return 'dog' if y_pred == 0 else 'cat'

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Binary Classification API!"

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    request_data = request.json
    idx = request_data["idx"]
    prediction = predict(model, x_test_preprocessed, idx)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":

    app.run(port=80)

