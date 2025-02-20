#This is a simple neural network that uses the iris dataset to predict the species of iris based on the sepal length, sepal width, petal length, and petal width.
#The model uses 3 layers, 4 input nodes, 8 hidden nodes, and 3 output nodes. The model is trained using the Adam optimizer and the CrossEntropyLoss function.
#Video here https://www.youtube.com/watch?v=JHWqWIoac2I
#iris dataset here: https://archive.ics.uci.edu/dataset/53/iris

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk

#build an ai model. derive it from nn.Module
#https://pytorch.org/docs/stable/generated/torch.nn.Module.html
#nn.Module is a fundamental base class in PyTorch used for building neural networks. It serves as a blueprint for defining custom neural network models by allowing you to encapsulate layers, parameters (like weights and biases), and the logic for forward propagation (how data flows through the network).
#Key Features:
#Modularity: You subclass nn.Module to create your own neural network architecture, organizing layers (e.g., nn.Linear, nn.Conv2d) and operations.
#Parameter Management: It automatically tracks all trainable parameters (e.g., weights) in the network, making them accessible for optimization.
#Forward Method: You define a forward() method in your subclass to specify how input data is processed through the network.
#Hierarchical Structure: It supports nesting modules (e.g., a model can contain submodules), enabling complex architectures.

class Model(nn.Module):
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1) #Layer 0 -The input layer of 4 input 'nodes'
    self.fc2 = nn.Linear(h1, h2) #Layer 1 - hidden layer
    self.out = nn.Linear(h2, out_features) #Layer #2 the output layer

  def forward(self, x):
    x = F.relu(self.fc1(x)) #call Layer 0, passing in a tensor full of x data
    x = F.relu(self.fc2(x)) #call Layer 1, with modified tensor
    x = self.out(x) #output layer
    return x #return the output layer tensor


#Get the data
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
df = pd.read_csv(url)

print("Sample of whats in df")
print(df.head())

#prepare the data

#convert the strings in the last col to int values. so 'Setosa' = 0 after the convert
df['species'] = pd.Categorical(df['species']).codes
print("Sample of whats in df['species']")
print(df['species'].head())

#make X a numpy array with the species col dropped from it
X = df.drop('species', axis=1).values

#make y a numpy array of only the species col
y = df['species'].values

#split X and y into train and test arrays
X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.2, random_state=32)

#convert Xtrain and Xtest to tensors of type float
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#convert ytrain and ytest to tensors of type int
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#set up the model
torch.manual_seed(32)
model = Model()

#Use the CrossEntropyLoss function in nn
#https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#nn.CrossEntropyLoss is a loss function in PyTorch commonly used for training neural networks in multi-class classification tasks. It combines log-softmax and negative log-likelihood (NLL) loss into a single operation, making it efficient and convenient for evaluating how well a model’s predicted class probabilities match the true labels.
criterion = nn.CrossEntropyLoss()

#set the optimizer and learning rates
#The Adam optimizer in PyTorch, implemented as torch.optim.Adam, is a popular optimization algorithm used to update the parameters (e.g., weights and biases) of a neural network during training. Adam stands for Adaptive Moment Estimation, and it combines the benefits of two other methods: momentum (using past gradients) and RMSProp (adapting learning rates based on gradient magnitudes). It’s widely used because it’s efficient, adaptive, and works well for a variety of deep learning tasks.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train the model
epochs = 100
losses = []

for i in range(epochs):
  i += 1
  y_pred = model.forward(X_train)
  loss = criterion(y_pred, y_train)
  losses.append(loss.detach().numpy())

  optimizer.zero_grad() #resets the gradients to 0
  loss.backward() #this is the backpropagation  
  optimizer.step() #updates the weights by the gradients found in backpropagation

#graph the loss per epoch
plt.plot(range(epochs), losses)
plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')

#test the model with the xtest data
with torch.no_grad():
  y_eval = model.forward(X_test)
  loss = criterion(y_eval, y_test)

print(f'Loss: {loss}')

plt.show()
