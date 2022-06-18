import numpy as np
import csv
from django.core.files import File
from django.core.files.base import ContentFile

############## Nural Network Modules ############################## 
class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights


# Linear modules
#
# Each linear module has a forward method that takes in a batch of
# activations A (from the previous layer) and returns
# a batch of pre-activations Z.
#
# Each linear module has a backward method that takes in dLdZ and
# returns dLdA. This module also computes and stores dLdW and dLdW0,
# the gradients with respect to the weights.
class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        self.A = A
        return np.dot(self.W.T, A) + self.W0  # (m x n)^T (m x b) = (n x b)

    def backward(self, dLdZ):  # dLdZ is (n x b), uses stored self.A
        self.dLdW = np.dot(self.A, dLdZ.T)                  # (m x n)
        self.dLdW0 = dLdZ.sum(axis=1).reshape((self.n, 1))  # (n x 1)
        return np.dot(self.W, dLdZ)                         # (m x b)

    def sgd_step(self, lrate):  # Gradient descent step
        self.W -= lrate*self.dLdW
        self.W0 -= lrate*self.dLdW0


# Activation modules
#
# Each activation module has a forward method that takes in a batch of
# pre-activations Z and returns a batch of activations A.
#
# Each activation module has a backward method that takes in dLdA and
# returns dLdZ, with the exception of SoftMax, where we assume dLdZ is
# passed in.
class Tanh(Module):            # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):    # Uses stored self.A
        return dLdA * (1.0 - (self.A ** 2))


class ReLU(Module):              # Layer activation
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):    # uses stored self.A
        return dLdA * (self.A != 0)


class SoftMax(Module):           # Output activation
    def forward(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def backward(self, dLdZ):    # Assume that dLdZ is passed in
        return dLdZ

    def class_fun(self, Ypred):  # Return class indices
        return np.argmax(Ypred, axis=0)


# Loss modules
#
# Each loss module has a forward method that takes in a batch of
# predictions Ypred (from the previous layer) and labels Y and returns
# a scalar loss value.
#
# The NLL module has a backward method that returns dLdZ, the gradient
# with respect to the preactivation to SoftMax (note: not the
# activation!), since we are always pairing SoftMax activation with
# NLL loss
class NLL(Module):       # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return float(np.sum(-Y * np.log(Ypred)))

    def backward(self):  # Use stored self.Ypred, self.Y
        return self.Ypred - self.Y


# Neural Network implementation
class Sequential:
    def __init__(self, modules, loss):            # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        sum_loss = 0
        for it in range(iters):
            if it%1000==0:
                print(it)
            i = np.random.randint(N)
            Xt = X[:, i:i+1]
            Yt = Y[:, i:i+1]
            Ypred = self.forward(Xt)
            sum_loss += self.loss.forward(Ypred, Yt)
            err = self.loss.backward()
            self.backward(err)
            self.sgd_step(lrate)

    def forward(self, Xt):                        # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def predict(self, Xt):                        # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return self.modules[-1].class_fun(Xt)

    def backward(self, delta):                    # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):                    # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints current loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '	Acc =', acc, '	Loss =', cur_loss)

    
    def get_accuracy(self, X, Y):
        D, N = X.shape
        i = np.random.randint(N)
        cf = self.modules[-1].class_fun
        acc = np.mean(cf(self.forward(X)) == cf(Y))
        return acc


nn_modules_dict = {'Linear':Linear,'ReLU':ReLU, 'SoftMax':SoftMax, 'Tanh':Tanh}


def create_nn_from_description(description, nn_modules = nn_modules_dict):
    layers = []
    for layer in description:
        if layer[0] == 'Linear':
            layers.append(nn_modules[layer[0]](int(layer[1]), int(layer[2])))
        else:
            layers.append(nn_modules[layer[0]]())
    return Sequential(layers, NLL())


def process_model_description(model_str):
    s = model_str.split(';')
    #remove the last element
    s.pop()
    print(s)
    processed_description = []
    for i in s:
        processed_description.append(i[1:-1].split(','))
    return processed_description