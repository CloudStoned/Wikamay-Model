import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

# Load data
with open('RFC_MODEL/data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Convert data to pandas DataFrame with named features
feature_names = [f"feature_{i}" for i in range(len(data_dict['data'][0]))]
X = pd.DataFrame(data_dict['data'], columns=feature_names)
y = pd.Series(data_dict['labels'], name='target')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Create a PMMLPipeline
pipeline = PMMLPipeline([
    ("classifier", model)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Export the model to PMML
sklearn2pmml(pipeline, "random_forest_model.pmml", with_repr=True)

print("Model exported as PMML")