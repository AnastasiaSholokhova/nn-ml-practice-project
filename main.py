"""Main file"""
import os
import data_load
import preprocess
import train
import eval
import pred

RELATIVE_PATH = "datasets\\csv\\"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

x_train_path = os.path.join(project_root, RELATIVE_PATH, 'input.csv')
y_train_path = os.path.join(project_root, RELATIVE_PATH,"labels.csv")
x_test_path = os.path.join(project_root, RELATIVE_PATH, 'input_test.csv')
y_test_path = os.path.join(project_root, RELATIVE_PATH, 'labels_test.csv')

x_train, y_train, x_test, y_test = data_load.data_load(x_train_path, y_train_path,
                                                       x_test_path, y_test_path)
x_train_processed, y_train_processed, x_test_processed, y_test_processed = preprocess.preprocess(x_train, y_train, x_test, y_test)
model = train.train_model(x_train_processed, y_train_processed)
metrics = eval.evaluate_model(model, x_test_processed, y_test_processed)
PREDICTION = pred.predict_image_class(model, x_test_processed)
print(PREDICTION)
