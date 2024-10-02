import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import subprocess
import sys

def show_data(data):
    print(data['labels'])
    print(data['data'])

def shape_data(data_dict):
    print("Data shape:", np.asarray(data_dict['data']).shape)
    print("Number of features:", np.asarray(data_dict['data']).shape[1])

def convert_to_joblib(model, filename='random_forest_model.joblib'):
    try:
        dump(model, filename)
        print(f"Model successfully saved in joblib format as: {filename}")
        return True
    except Exception as e:
        print(f"Error saving model to joblib format: {e}")
        return False

def train(data_dict):
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    print("Data shape:", data.shape)
    n_features = data.shape[1]
    print("Number of features:", n_features)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    # Convert and save the model using joblib
    if convert_to_joblib(model):
        print("Model saved in joblib format.")
    else:
        print("Failed to save model in joblib format.")

    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    converted_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model
    onnx_model_path = "sklearn_model.onnx"
    with open(onnx_model_path, "wb") as f:
        f.write(converted_model.SerializeToString())
    
    print(f"ONNX model saved as: {onnx_model_path}")

    # Convert the ONNX model to ORT format using subprocess
    output_dir = "."  # Current directory
    try:
        result = subprocess.run(
            [sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", 
             onnx_model_path, 
             "--output_dir", output_dir],
            check=True,
            capture_output=True,
            text=True
        )
        ort_model_path = os.path.join(output_dir, os.path.splitext(os.path.basename(onnx_model_path))[0] + ".ort")
        print(f"ONNX model successfully converted to ORT format: {ort_model_path}")
        print(f"Conversion output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting ONNX model to ORT format: {e}")
        print(f"Error output: {e.stderr}")

if __name__ == '__main__':
    data_dict = load(r'D:\SignLanguage\RFC_MODEL\data.pickle')
    show_data(data_dict)
    shape_data(data_dict)
    # train(data_dict)