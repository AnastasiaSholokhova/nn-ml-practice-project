def predict_image_class(model, X_test_processed):
    idx = int(input("Enter the index of the image you want to predict (0 to {}): ".format(len(X_test_processed)-1)))
    y_pred = model.predict(X_test_processed[idx].reshape(1, 100, 100, 3))
    y_pred = y_pred > 0.5
    if y_pred == 0:
        pred = 'dog'
    else:
        pred = 'cat'
    return pred