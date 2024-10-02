import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from huggingface_hub import push_to_hub_sklearn

import os
from dotenv import load_dotenv

load_dotenv()

HUB_API = os.getenv('HUB_API')

def show_data(data):
    print(data['labels'])
    print(data['data'])

def train(data_dict):
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    # Save model locally
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved locally")

    return model

def push_to_hub(model, repo_name, token):
    try:
        push_to_hub_sklearn(
            model,
            repo_name=repo_name,
            token=token
        )
        print(f"Model successfully pushed to Hugging Face Hub: {repo_name}")
    except Exception as e:
        print(f"An error occurred while pushing to Hugging Face Hub: {str(e)}")

if __name__ == '__main__':
    data_dict = pickle.load(open('data.pickle', 'rb'))
    # show_data(data_dict)
    trained_model = train(data_dict)
    
    # Push model to Hugging Face Hub
    repo_name = "cLoudstone99/ASL_RECOG"  # Updated with the provided repository name
    push_to_hub(trained_model, repo_name, HUB_API)