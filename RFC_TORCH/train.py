import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

class RandomForest(nn.Module):
    def __init__(self, input_dim, num_classes, num_trees=100, tree_depth=5):
        super(RandomForest, self).__init__()
        self.trees = nn.ModuleList([DecisionTree(input_dim, num_classes, tree_depth) for _ in range(num_trees)])

    def forward(self, x):
        tree_preds = torch.stack([tree(x) for tree in self.trees])
        return torch.mean(tree_preds, dim=0)

class DecisionTree(nn.Module):
    def __init__(self, input_dim, num_classes, depth):
        super(DecisionTree, self).__init__()
        self.depth = depth
        self.decision_nodes = nn.ModuleList([DecisionNode(input_dim) for _ in range(2**depth - 1)])
        self.leaf_nodes = nn.Parameter(torch.randn(2**depth, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        for d in range(self.depth):
            decisions = self.decision_nodes[indices](x)
            indices = 2 * indices + (decisions <= 0.5).long()

        return self.leaf_nodes[indices]

class DecisionNode(nn.Module):
    def __init__(self, input_dim):
        super(DecisionNode, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.weight) + self.bias)

def train(data_dict):
    # Convert string labels to numerical labels
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(data_dict['labels'])
    
    data = torch.tensor(data_dict['data'], dtype=torch.float32)
    labels = torch.tensor(numerical_labels, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = data.shape[1]
    num_classes = len(label_encoder.classes_)
    model = RandomForest(input_dim, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = accuracy_score(y_test, predicted)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        _, predicted = torch.max(y_predict, 1)
        accuracy = accuracy_score(y_test, predicted)
        print(f'Final Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model for Android (TorchScript format)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, 'model.pt')
    print("Model saved in TorchScript format")

    # Save the LabelEncoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("LabelEncoder saved")

if __name__ == '__main__':
    with open('RFC_MODEL/data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    train(data_dict)