import sklearn

def predict(input_data, model):
    """
    Predicts the output based on the given input data and model.

    Parameters:
    - input_data: Data to be predicted on.
    - model: Trained model to use for prediction.

    Returns:
    - Predicted output.
    """
    return model.predict(input_data)
