import numpy as np
import matplotlib.pyplot as plt

class shallow():
    def __init__(self, layer, learning_rate=0.01, momentum=0.9, activative="relu"):
        self.layer = layer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activative = activative
        self.weights, self.biases = self.init_params()

    def activation(self, x):
        if self.activative == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activative == "relu":
            return np.maximum(0, x)
        elif self.activative == "tanh":
            return np.tanh(x)
        elif self.activative == "linear":
            return x

    def activation_diff(self, x):
        if self.activative == "sigmoid":
            return x * (1 - x)
        elif self.activative == "relu":
            return np.where(x > 0, 1.0, 0.0)
        elif self.activative == "tanh":
            return 1 - np.square(x)
        elif self.activative == "linear":
            return np.ones_like(x)

    def init_params(self):
        weights = [np.random.randn(self.layer[i], self.layer[i-1]) * 0.01 for i in range(1, len(self.layer))]
        biases = [np.random.rand(self.layer[i], 1) for i in range(1, len(self.layer))]
        return weights, biases

    def forward_pass(self, input_data):
        self.x = [input_data.T]
        for i in range(len(self.layer) - 1):
            self.x.append(self.activation((np.dot(self.weights[i], self.x[i])))+ self.biases[i])
        return self.x[-1]

    def backward_pass(self, input_data, output_data):
        m = input_data.shape[0]
        delta_weights = []
        delta_biases = []
        gradients = []
        err = (output_data - self.x[-1]) / m if self.layer[0]==8 else (output_data.T - self.x[-1]) / m
        for i in reversed(range(len(self.layer) - 1)):
            dG = err * self.activation_diff(self.x[i+1])
            gradients.append(dG)
            delta_weights.append(np.dot(dG, self.x[i].T))
            delta_biases.append(np.sum(dG, axis=1, keepdims=True))
            if i > 0:
                err = np.dot(self.weights[i].T, dG)
        gradients.reverse()
        delta_weights.reverse()
        delta_biases.reverse()
        return delta_weights, delta_biases

    def update_parameters(self, delta_weights, delta_biases):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * delta_weights[i]
            self.biases[i] -= self.learning_rate * delta_biases[i]

    def compute_loss(self, predictions, output_data):
        return np.mean(np.square(predictions - output_data if self.layer[0]==8 else output_data.T))

    def train(self, input_data, output_data, val_input, val_output, epochs=1000):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward_pass(input_data)
            train_loss = self.compute_loss(predictions, output_data)
            train_losses.append(train_loss)

            # Backward pass
            delta_weights, delta_biases = self.backward_pass(input_data, output_data)
            self.update_parameters(delta_weights, delta_biases)

            # Validation
            val_predictions = self.forward_pass(val_input)
            val_loss = self.compute_loss(val_predictions, val_output)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                
        print("Training complete.")
        plt.plot(train_losses, label='Flood Training Loss' if self.layer[0]==8 else 'Crooss Training Loss')
        plt.plot(val_losses, label='Flood Validation Loss' if self.layer[0]==8 else 'Crooss Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

def readfile(filename):
    return readfile_1() if filename=="flood" else (readfile_2() if filename=="cross" else print("Error file"))

def readfile_1(filename = 'Flood_dataset.txt'):
    data = []
    input_data = []
    design_output = []
    with open(filename) as f:
        for line in f.readlines()[2:]:
            data.append([float(element[:-1]) for element in line.split()])
    data = np.array(data)
    np.random.shuffle(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    data = (data - min_vals) / (max_vals - min_vals)
    for i in data:
        input_data.append(i[:-1])
        design_output.append(np.array(i[-1]))
    return input_data, design_output
    
def readfile_2(filename = 'cross.txt'):
    data = []
    input_data = []
    design_output = []
    with open(filename) as f:
        a = f.readlines()
        for line in range(1, len(a), 3):
            z = np.array([float(element) for element in a[line][:-1].split()])
            zz = np.array([float(element) for element in a[line+1].split()])
            data.append(np.append(z, zz))
    data = np.array(data)
    np.random.shuffle(data)
    for i in data:
        input_data.append(i[:-2])
        design_output.append(i[-2:])
    return input_data, design_output

def split_data(input_data, output_data, val_ratio=0.2):
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)
    input_data = input_data[indices]
    output_data = output_data[indices]
    
    split_point = int((1 - val_ratio) * len(input_data))
    X_train = input_data[:split_point]
    Y_train = output_data[:split_point]
    X_val = input_data[split_point:]
    Y_val = output_data[split_point:]
    
    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__":
    ### Flood_dataset file must has layer format by [8, ... , 1]
    ### cross_dataset file must has layer format by [2, ... , 2]

    X, Y = readfile('flood')
    X_train, X_val, Y_train, Y_val = split_data(X, Y, val_ratio=0.2)
    nn = shallow(layer=[8, 16, 1], learning_rate=0.01, activative="sigmoid")
    nn.train(np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), epochs=500)
    
    X1, Y1 = readfile('cross')
    X1_train, X1_val, Y1_train, Y1_val = split_data(X1, Y1, val_ratio=0.2)
    nn1 = shallow(layer=[2, 16, 2], learning_rate=0.01, activative="relu")
    nn1.train(np.array(X1_train), np.array(Y1_train), np.array(X1_val), np.array(Y1_val), epochs=500)
    
    plt.show()
