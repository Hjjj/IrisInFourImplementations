# This is a simple neural network that uses the iris dataset to predict the species of iris based on the sepal length, sepal width, petal length, and petal width.
# The model uses 3 layers, 4 input nodes, 8 hidden nodes, and 3 output nodes. The model is trained using the Adam optimizer and the SparseCategoricalCrossentropy loss function.
# Video here https://www.youtube.com/watch?v=JHWqWIoac2I
# iris dataset here: https://archive.ics.uci.edu/dataset/53/iris

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk

# Define the model using TensorFlow's Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),  # Layer 0 - The input layer of 4 input 'nodes'
    tf.keras.layers.Dense(9, activation='relu'),  # Layer 1 - hidden layer
    tf.keras.layers.Dense(3)  # Layer 2 - the output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Get the data
url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
df = pd.read_csv(url)

print("Sample of what's in df")
print(df.head())

# Prepare the data
df['species'] = pd.Categorical(df['species']).codes
print("Sample of what's in df['species']")
print(df['species'].head())

X = df.drop('species', axis=1).values
y = df['species'].values

# Split X and y into train and test arrays
X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.2, random_state=32)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Graph the loss per epoch
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Test the model with the X_test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

plt.show()
