from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle

# Define a mapping from integer labels to actual class names
class_names = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
}

def train(data_dict):
    # Convert data and labels to numpy arrays
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # Get unique labels in the dataset
    unique_labels = np.unique(labels)

    # Ensure unique_labels are integers for indexing the dictionary
    unique_labels = unique_labels.astype(int)

    # Filter the class names based on the unique labels
    filtered_class_names = [class_names[label] for label in unique_labels]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Initialize the RandomForest model
    model = RandomForestClassifier()

    # Train the model
    model.fit(x_train, y_train)

    # Predict on the test set
    y_predict = model.predict(x_test)

    # Calculate the accuracy score
    score = accuracy_score(y_predict, y_test)

    # Generate and print the classification report with filtered class names
    print("\nClassification Report:")
    print(classification_report(y_test, y_predict, target_names=filtered_class_names))
    print('{}% of samples were classified correctly!'.format(score * 100))

    # Save the trained model
    # model_name = 'alph_model'
    # with open(f'{model_name}.p', 'wb') as f:
    #     pickle.dump({'model': model}, f)
    # print("Model Created")


if __name__ == '__main__':
    # Load the data dictionary
    data_dict = pickle.load(open('./pickle/NUM_DATA.pickle', 'rb'))

    # Uncomment the line below to see the data structure
    # show_data(data_dict)

    # Train the model and evaluate
    train(data_dict)
