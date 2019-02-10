import numpy as np


class Perceptron:

    def __init__(self, data, answers, learning_rate: float):
        self.lr = learning_rate
        self.input = data
        self.answers = answers
        self.weights = dict()
        self.neurons = dict()
        self.errors = dict()
        self.w_bias = dict()
        self.mse_for_plot = np.array([])
        self.mse = 0
        self.idx = 0

    @staticmethod
    def f(x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    @staticmethod
    def d(x):
        return (1.0 - x) * x

    def add_layer(self, input_neurons: int, out_neurons: int):
        self.weights[self.idx] = np.random.sample([input_neurons, out_neurons])
        self.w_bias[self.idx] = np.random.sample([1, out_neurons])
        self.neurons[self.idx] = np.array([])
        self.errors[self.idx] = np.array([])
        self.idx += 1

    def forward_propagation(self, input_x):
        x = input_x

        for k in range(self.idx):
            w = self.weights[k]  # matrix
            b = self.w_bias[k]  # vector
            self.neurons[k] = self.f(np.dot(x, w) + b)
            x = self.neurons[k]

        return x

    def find_errors(self, y_true):
        idx = self.idx - 1

        for i in range(idx, -1, -1):
            if i == idx:
                n = self.neurons[i]
                self.mse += np.sum((y_true - n) ** 2)
                self.errors[i] = (y_true - n) * (self.d(n))
            else:
                n = self.neurons[i]
                w = self.weights[i+1]
                delta = self.errors[i+1]
                self.errors[i] = np.multiply(self.d(n), (np.dot(delta, w.T)))

        return self.errors

    def update_weights(self, input_x):

        for l in range(self.idx):
            delta = self.errors[l]
            gradient = np.dot(input_x.T, delta)
            self.weights[l] += self.lr * gradient
            self.w_bias[l] += self.lr * delta
            input_x = self.neurons[l]

        for i in range(self.idx):
            self.neurons[i] = np.array([])
            self.errors[i] = np.array([])

    def train(self, epoch: int):
        for e in range(epoch):
            for i in range(self.input.shape[0]):
                df_x = self.input[i][np.newaxis, :]
                df_y = self.answers[i][np.newaxis, :]
                self.forward_propagation(df_x)
                self.find_errors(df_y)
                self.update_weights(df_x)
            print("EPOCH - {}, MSE - {}".format(e, self.mse / self.input.shape[0]))
            self.mse_for_plot = np.append(self.mse_for_plot, self.mse / self.input.shape[0])
            self.mse = 0
        return self

    def predict(self, x_test):
        y = np.zeros([x_test.shape[0], 1])
        for i in range(x_test.shape[0]):
            df_x = x_test[i][np.newaxis, :]
            out = self.forward_propagation(df_x)
            answer = np.argmax(out)
            y[i] = answer
        return y
