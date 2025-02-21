# This is a simple neural network that uses the iris dataset to predict the species of iris based on the sepal length, sepal width, petal length, and petal width.
# The model uses 3 layers, 4 input nodes, 8 hidden nodes, and 3 output nodes. The model is trained using gradient descent and the CrossEntropyLoss function.
# Video here https://www.youtube.com/watch?v=JHWqWIoac2I
# iris dataset here: https://archive.ics.uci.edu/dataset/53/iris


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk

class Model:
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        self.fc1_weights = np.random.randn(in_features, h1)
        self.fc1_bias = np.zeros(h1)
        self.fc2_weights = np.random.randn(h1, h2)
        self.fc2_bias = np.zeros(h2)
        self.out_weights = np.random.randn(h2, out_features)
        self.out_bias = np.zeros(out_features)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.fc1_weights) + self.fc1_bias
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.fc2_weights) + self.fc2_bias
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.out_weights) + self.out_bias
        return self.softmax(self.z3)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, x, y_true, y_pred, learning_rate=0.01):
        m = y_true.shape[0]
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(m), y_true] = 1

        dz3 = y_pred - y_true_one_hot
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0) / m

        dz2 = np.dot(dz3, self.out_weights.T) * (self.a2 > 0)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        dz1 = np.dot(dz2, self.fc2_weights.T) * (self.a1 > 0)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        self.out_weights -= learning_rate * dw3
        self.out_bias -= learning_rate * db3
        self.fc2_weights -= learning_rate * dw2
        self.fc2_bias -= learning_rate * db2
        self.fc1_weights -= learning_rate * dw1
        self.fc1_bias -= learning_rate * db1

# Get the data
url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
df = pd.read_csv(url)

print("Sample of whats in df")
print(df.head())

# Prepare the data
df['species'] = pd.Categorical(df['species']).codes
print("Sample of whats in df['species']")
print(df['species'].head())

X = df.drop('species', axis=1).values
y = df['species'].values

X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.2, random_state=32)

# Set up the model
np.random.seed(32)
model = Model()

# Train the model
epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = model.compute_loss(y_pred, y_train)
    losses.append(loss)
    model.backward(X_train, y_train, y_pred)

# Graph the loss per epoch
plt.plot(range(epochs), losses)
plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Test the model with the X_test data
y_eval = model.forward(X_test)
loss = model.compute_loss(y_eval, y_test)

print(f'Loss: {loss}')

plt.show()