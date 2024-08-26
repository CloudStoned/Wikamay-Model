import numpy as np
from sklearn.metrics import accuracy_score
import onnxruntime as ort
from joblib import load
import os 

def test_ort_model(ort_model_path, data_dict):
    # Load the ORT model
    ort_session = ort.InferenceSession(ort_model_path)

    # Prepare the test data
    data = np.asarray(data_dict['data']).astype(np.float32)
    labels = np.asarray(data_dict['labels'])

    # Get the input name of the model
    input_name = ort_session.get_inputs()[0].name

    # Run inference
    ort_predictions = ort_session.run(None, {input_name: data})[0]

    # Calculate accuracy
    ort_accuracy = accuracy_score(labels, ort_predictions)
    print(f"ORT Model Accuracy: {ort_accuracy * 100:.2f}%")

    return ort_accuracy


def test_joblib_model(joblib_model_path, data_dict):
    # Load the joblib model
    model = load(joblib_model_path)

    # Prepare the test data
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # Make predictions
    joblib_predictions = model.predict(data)

    # Calculate accuracy
    joblib_accuracy = accuracy_score(labels, joblib_predictions)
    print(f"Joblib Model Accuracy: {joblib_accuracy * 100:.2f}%")

    return joblib_accuracy

if __name__ == '__main__':
    data_dict = load(r'D:\SignLanguage\RFC_MODEL\data.pickle')
    
    ort_model_path = r'D:\SignLanguage\RFC_ONNX\sklearn_model.ort'
    ort_accuracy = test_ort_model(ort_model_path, data_dict)

    joblib_model_path = "RFC_ONNX/random_forest_model.joblib"
    joblib_accuracy = test_joblib_model(joblib_model_path, data_dict)