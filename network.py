import random
import numpy as np

class Network(object):
    """a recurrent neural network"""

    def __init__(self, wordsize):
        # initializes two wordsize by wordsize weight arrays with Gaussian distributions N(0, 0.005) clipped to the range [-0.01, 0.01]
        self.hidden_weight, self.input_weight = tuple(np.clip(np.random.normal(0, 0.005, (wordsize, wordsize)), -0.01, 0.01) for i in range(2))

        # the bias is initialized to the 0 vector
        self.hidden_bias = np.zeros((wordsize, 1))

    '''def feedforward(self, a, h=None, i=0):
        # where a is the list of input vectors, h is the previous hidden layer's output, and i is the number of the current iteration
        if (h is None): h = np.zeros((self.wordsize, 1)) # initialize h0 to the zero vector
        if (i < len(a)):
            # feed the output to the next layer
            return self.feedforward(a, sigmoid(self.hidden_weight.dot(h) + self.input_weight.dot(a[i]) + self.hidden_bias), i+1)
        else:
            # return the final hidden vector, hn
            return h'''

    def feedforward(self, x):
        # where x is the list of input vectors
        s = [None]*len(x) # list to store the hidden s vectors (pre-activation/sigmoid)
        h = [None]*len(x) # list to store the hidden h vectors(post/activation/sigmoid)

        s[0] = self.input_weight.dot(x[0]) + self.hidden_bias
        h[0] = sigmoid(s[0])

        for i in range(1, len(x)):
            s[i] = self.hidden_weight.dot(h[i-1]) + self.input_weight.dot(x[i]) + self.hidden_bias # feed to next layer
            h[i] = sigmoid(s[i]) # activation function

        return s, h

    def backprop(self, x, s, h, nabla_start, truncate):
        # where x, s and h are same as above, nabla_start is the gradient of C with respect to the final h
        n = len(s) # number of layers

        # arrays to store grad(C) wrt hidden_weight, input_weight, and hidden_bias
        nabla_U = np.zeros(self.hidden_weight.shape)
        nabla_W = np.zeros(self.input_weight.shape)
        nabla_b = np.zeros(self.hidden_bias.shape)

        # delta is the gradient of C with respect to the s vector of the current layer
        delta = np.multiply(nabla_start, sigmoid_prime(s[n-1])) # start with delta of the final layer

        for i in range(n-1, max(-1, n-truncate-1), -1): # sum gradients either all the way to layer 0 or truncate layers back; truncate must be at least 1
            if i > 0: nabla_U += np.outer(delta, h[i-1]) # add grad(C) wrt hidden_weight for layer i
            nabla_W += np.outer(delta, x[i])             # add grad(C) wrt input_weight for layer i
            nabla_b += delta                             # add grad(C) wrt hidden_bias for layer i
            delta = np.multiply((self.hidden_weight.T.dot(delta)), sigmoid_prime(s[i])) # find delta for s[i-1]

        return nabla_U, nabla_W, nabla_b # grad(C) wrt hidden_weight, input_weight, hidden_bias

class DoubleNetwork(object):
    """combination of two neural networks that will take the output of each network, concatenate them, apply a final weight and bias, and transform the result using softmax
    in order to output a distribution representative of the probability of the two network inputs being the a given class"""
    def __init__(self, wordsize, num_classes):
        self.net1 = Network(wordsize)
        self.net2 = Network(wordsize)

        # a matrix that will transform the concatenation of the two network outputs into a vector with dimension equal to the number of classes
        self.final_weight = np.clip(np.random.normal(0, 0.005, (num_classes, 2*wordsize)), -0.01, 0.01) # initialized to a Gaussian distribution N(0, 0.005) clipped to [-0.01, 0.01]

        self.final_bias = np.zeros((num_classes, 1))

    def feedforward(self, x1, x2):
        # where x1 and x2 are lists of vectors that are the inputs for each network
        s1, h1 = self.net1.feedforward(x1)
        s2, h2 = self.net2.feedforward(x2) 
        h = np.concatenate((h1[-1], h2[-1])) # output of two networks combined

        p = softmax(np.dot(self.final_weight, h) + self.final_bias) # final output probability distribution
        return s1, s2, h1, h2, h, p

    def backprop(self, x1, x2, s1, s2, h1, h2, h, p, target, truncate):
        # where target is a one-hot column vector representing the correct class
        nabla_f = p - target # grad(C) wrt final bias
        nabla_V = np.outer(nabla_f, h) # grad(C) wrt final weight
        nabla_h = self.final_weight.T.dot(nabla_f) # grad(C) wrt final hidden vector, only used as starting point for backprop of each network

        nabla_U1, nabla_W1, nabla_b1 = self.net1.backprop(x1, s1, h1, np.split(nabla_h, 2)[0], truncate)
        nabla_U2, nabla_W2, nabla_b2 = self.net2.backprop(x2, s2, h2, np.split(nabla_h, 2)[1], truncate)

        return nabla_U1, nabla_W1, nabla_b1, nabla_U2, nabla_W2, nabla_b2, nabla_V, nabla_f

    def SGD(self, data, batch_size, eta, epochs, truncate, test_data=None):
        """where data is a list of tuples in the form (x1, x2, target).
        test_data is an optional single tuple in the form (x1, x2) and if included,
        will be evaluated and the result p printed after each epoch."""

        for i in range(epochs):
            random.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)] # partitions data into batches of batch_size, truncating the last batch at the end of the data
            for batch in batches:
                self.train_batch(batch, eta, truncate)

            if test_data:
                print("Done epoch {0}. Test data result:\n{1}".format(i, self.test_eval(test_data[0], test_data[1])))
            else:
                print("Done epoch %d" % i)

    def train_batch(self, batch, eta, truncate):
        nabla_U1 = nabla_W1 = np.zeros(self.net1.hidden_weight.shape) 
        nabla_U2 = nabla_W2 = np.zeros(self.net2.hidden_weight.shape)
        nabla_b1 = np.zeros(self.net1.hidden_bias.shape)
        nabla_b2 = np.zeros(self.net2.hidden_bias.shape)
        nabla_V = np.zeros(self.final_weight.shape)
        nabla_f = np.zeros(self.final_bias.shape)
        for x1, x2, target in batch:
            s1, s2, h1, h2, h, p = self.feedforward(x1, x2)
            # adds the result of backprop to every nabla_
            nabla_U1, nabla_W1, nabla_b1, nabla_U2, nabla_W2, nabla_b2, nabla_V, nabla_f = tuple(sum(x) for x in zip(
                (nabla_U1, nabla_W1, nabla_b1, nabla_U2, nabla_W2, nabla_b2, nabla_V, nabla_f), 
                self.backprop(x1, x2, s1, s2, h1, h2, h, p, target, truncate)))

        self.net1.hidden_weight -= nabla_U1 * eta
        self.net1.input_weight -= nabla_W1 * eta
        self.net1.hidden_bias -= nabla_b1 * eta

        self.net2.hidden_weight -= nabla_U2 * eta
        self.net2.input_weight -= nabla_W2 * eta
        self.net2.hidden_bias -= nabla_b2 * eta

        self.final_weight -= nabla_V * eta
        self.final_bias -= nabla_f * eta

    def test_eval(self, x1, x2):
        s1, s2, h1, h2, h, p = self.feedforward(x1, x2)
        return p

       

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cost(p, y):
    """computes cross-entropy loss for a given network output and correct assignment.
    p is the distribution of probabilities output by the network, y is a one-hot vector corresponding to the correct class."""
    return -1 * np.asscalar(np.log(p.tolist()[y.tolist().index([1])])) # -ln(p[j]) where j is the correct class
 
"""net = DoubleNetwork(10, 6)
sentence1 = [np.random.normal(0, 1, (10, 1)) for i in range(7)]
sentence2 = [np.random.normal(0, 1, (10, 1)) for i in range(4)]
targ = np.array([0, 0, 0, 1, 0, 0])
targ = np.reshape(targ, (6, 1))

s1, s2, h1, h2, h, p = net.feedforward(sentence1, sentence2)

U1, W1, b1, U2, W2, b2, V, f = net.backprop(sentence1, sentence2, s1, s2, h1, h2, h, p, targ, 100)
"""
