from joblib import load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

# Load the joblib model
model = load('model.joblib')

# Get the number of features (assuming it's the same as the input dimension)
# For scikit-learn 1.2.2, we use n_features_in_ if available, otherwise fall back to n_features_
n_features = getattr(model, 'n_features_in_', getattr(model, 'n_features_', None))

if n_features is None:
    # If n_features is still None, we need to infer it from the model
    # This might depend on your specific model type
    raise ValueError("Could not determine the number of input features. Please specify manually.")

# Create the ONNX model
initial_type = [('float_input', FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved")