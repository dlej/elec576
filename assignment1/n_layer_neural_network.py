__author__ = 'Daniel LeJeune'

from itertools import product

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DeepNeuralNetwork(object):

    def __init__(self, input_dim, layer_specs, loss=None, reg_penalty=1e-2, seed=0):
        '''

        :param input_dim:
        :param layer_specs: list [(output_width, ActivationFunction)] of specs for each layer
        :param loss:
        :param reg_penalty:
        :param seed:
        '''

        np.random.seed(seed)
        self.input_dim = input_dim

        self.layers = []
        for output_dim, activation in layer_specs:
            self.layers.append(Layer(input_dim=input_dim, output_dim=output_dim, activation=activation))
            input_dim = output_dim

        if loss is None:
            self.loss = Losses.CrossEntropy()
        else:
            self.loss = loss

        self.reg_penalty = reg_penalty
        self._needs_feedforward = True

    def feedforward(self, X):

        for layer in self.layers:
            X = layer.feedforward(X)

        self.output = X
        self._needs_feedforward = False

        return self.output

    def backprop(self, y):

        self.loss.compute(self.output, y)

        grad = self.loss.grad
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)

    def update(self, step_size):

        for layer in self.layers:
            layer.update(step_size, self.reg_penalty)

        self._needs_feedforward = True

    def compute_loss(self, X, y):

        if self._needs_feedforward:
            self.feedforward(X)

        loss = self.loss.compute(self.output, y)

        for layer in self.layers:
            loss += self.reg_penalty / 2 * np.sum(layer.W ** 2)

        return loss

    def train(self, X, y, step_size=1e-2, epochs=1000, verbose=False):

        for i in range(epochs):

            if self._needs_feedforward:
                self.feedforward(X)
            self.backprop(y)
            self.update(step_size)

            if verbose and i % 1000 == 0:
                print('Loss after iteration %d: %f' % (i, self.compute_loss(X, y)))

    def check_gradients(self, X, y, epsilon=1e-6):

        self.feedforward(X)
        self.backprop(y)

        for k, layer in enumerate(self.layers):

            W = layer.W
            dW = layer.dW

            dW_hat = np.zeros_like(W)

            for i, j in product(*(range(x) for x in W.shape)):

                delta = np.zeros_like(W)
                delta[i, j] = epsilon

                layer.W = W + delta
                self._needs_feedforward = True
                L_plus = self.compute_loss(X, y)

                layer.W = W - delta
                self._needs_feedforward = True
                L_minus = self.compute_loss(X, y)

                dW_hat[i, j] = (L_plus - L_minus) / 2 / epsilon

            layer.W = W
            dW = dW + self.reg_penalty * W

            print('Layer %d, max dW error: %f' % (k, np.max(np.abs((dW_hat - dW) / np.maximum(np.abs(dW_hat), np.abs(dW))))))

            b = layer.b

            db_hat = np.zeros_like(b)

            for i, j in product(*(range(x) for x in b.shape)):

                delta = np.zeros_like(b)
                delta[i, j] = epsilon

                layer.b = b + delta
                self._needs_feedforward = True
                L_plus = self.compute_loss(X, y)

                layer.b = b - delta
                self._needs_feedforward = True
                L_minus = self.compute_loss(X, y)

                db_hat[i, j] = (L_plus - L_minus) / 2 / epsilon

            layer.b = b

            print('Layer %d, max db error: %f' % (k, np.max(np.abs((db_hat - layer.db) / np.maximum(np.abs(db_hat), np.abs(layer.db))))))


class Layer(object):

    def __init__(self, input_dim, output_dim, activation=None):

        self.input_dim = input_dim
        self.output_dim = output_dim

        if activation is None:
            self.activation = ActivationFunctions.Tanh()
        else:
            self.activation = activation

        self.W = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
        self.b = np.zeros((output_dim, 1))

    def feedforward(self, X):

        self.input = X
        self.Z = X @ self.W.T + self.b.T
        self.output = self.activation.feedforward(self.Z)

        return self.output

    def backprop(self, doutput):

        self.dz = np.einsum('ijk,ik->ij', self.activation.grad, doutput)
        self.dinput = np.einsum('ijk,ik->ij', self.W.T[None, :, :], self.dz)

        self.dW = self.dz.T @ self.input
        self.db = self.dz.sum(0)[:, None]

        return self.dinput

    def update(self, step_size, reg_penalty=0):

        dW = self.dW + reg_penalty * self.W

        self.W -= step_size*dW
        self.b -= step_size*self.db


class ActivationFunctions:

    class Sigmoid(object):

        def feedforward(self, X):

            n_samples, n_features = X.shape

            self.output = 1 / (1 + np.exp(-X))

            self.grad = np.zeros((n_samples, n_features, n_features))
            for i in range(n_samples):
                self.grad[i, np.arange(n_features), np.arange(n_features)] = self.output[i, :] * (1 - self.output[i, :])

            return self.output

    class Tanh(object):

        def __init__(self):
            self.sigmoid = ActivationFunctions.Sigmoid()

        def feedforward(self, X):

            self.sigmoid.feedforward(2 * X)
            self.output = 2*self.sigmoid.output - 1
            self.grad = 4*self.sigmoid.grad

            return self.output

    class ReLU(object):

        def feedforward(self, X):

            n_samples, n_features = X.shape

            grad = (X > 0).astype(float)

            self.grad = np.zeros((n_samples, n_features, n_features))
            for i in range(n_samples):
                self.grad[i, np.arange(n_features), np.arange(n_features)] = grad[i, :]

            self.output = grad * X

            return self.output

    class Softmax(object):

        def feedforward(self, X):

            n_samples, n_features = X.shape

            self.output = np.exp(X)
            self.output /= self.output.sum(1)[:, None]

            self.grad = np.zeros((n_samples, n_features, n_features))
            for i in range(n_samples):
                self.grad[i, np.arange(n_features), np.arange(n_features)] = self.output[i, :]
                self.grad[i, :, :] -= np.outer(self.output[i, :], self.output[i, :])

            return self.output


class Losses:

    class CrossEntropy(object):

        def __init__(self):
            self.onehot_encoder = None

        def compute(self, X, y):

            n_samples, n_features = X.shape

            if np.ndim(y) == 1:
                y = y[:, None]

            if self.onehot_encoder is None:
                self.onehot_encoder = OneHotEncoder(n_features, sparse=False)
                y = self.onehot_encoder.fit_transform(y)
            else:
                y = self.onehot_encoder.transform(y)

            self.output = - np.sum(np.log(X) * y) / n_samples
            self.grad = - y / X / n_samples

            return self.output