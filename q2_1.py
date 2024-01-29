import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def read_mnist_images(images_path):
    with gzip.open(images_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        images_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images_data.reshape(num_images, -1)

def read_mnist_labels(file_path, num_labels):
    with gzip.open(file_path, 'r') as f:
        f.read(8)
        labels = np.frombuffer(f.read(1 * num_labels), dtype=np.uint8).astype(np.int64)
    return labels

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_neurons, output_size):
        self.input_size = input_size
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_size = output_size

        self.b1 = np.zeros((1, self.hidden_layer_neurons))
        self.b2 = np.zeros((1, self.output_size))
        self.W1 = np.random.randn(self.input_size, self.hidden_layer_neurons) * np.sqrt(2 / (self.input_size + self.hidden_layer_neurons))
        self.W2 = np.random.randn(self.hidden_layer_neurons, self.output_size) * np.sqrt(2 / (self.hidden_layer_neurons + self.output_size))

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.W1) + self.b1
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output = np.dot(self.hidden_layer_output, self.W2) + self.b2
        return self.output, self.hidden_layer_output

    def backward(self, X, y, learning_rate):
        grad_output = mse_loss_derivative(y, self.output)
        grad_W2 = np.dot(self.hidden_layer_output.T, grad_output)
        grad_b2 = np.sum(grad_output, axis=0, keepdims=True)
        grad_hidden = np.dot(grad_output, self.W2.T) * sigmoid_derivative(self.hidden_layer_input)
        grad_W1 = np.dot(X.T, grad_hidden)
        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.W1) + self.b1
        hidden_layer_output = sigmoid(hidden_layer_input)
        output = np.dot(hidden_layer_output, self.W2) + self.b2
        return np.argmax(output)

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            output, _ = self.forward(X)
            loss = mse_loss(y, output)
            losses.append(loss)
            self.backward(X, y, learning_rate)
            print(f'Epoch {epoch}, Loss: {loss}')

        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

train_images_mnist = read_mnist_images('mnist/train-images-idx3-ubyte.gz')
train_labels_mnist = read_mnist_labels('mnist/train-labels-idx1-ubyte.gz', 60000)
test_images_mnist = read_mnist_images('mnist/t10k-images-idx3-ubyte.gz')
test_labels_mnist = read_mnist_labels('mnist/t10k-labels-idx1-ubyte.gz', 10000)

num_classes = 10
train_labels_one_hot = np.eye(num_classes)[train_labels_mnist]

input_size = train_images_mnist.shape[1]
hidden_layer_neurons = 64
output_size = num_classes
learning_rate = 0.01
epochs = 400

model = NeuralNetwork(input_size, hidden_layer_neurons, output_size)
model.train(train_images_mnist, train_labels_one_hot, epochs, learning_rate)

predictions = [model.predict(img) for img in test_images_mnist]

accuracy = accuracy_score(test_labels_mnist, predictions)
print(f'Test Accuracy: {accuracy}')
