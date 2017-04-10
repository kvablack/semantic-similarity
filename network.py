import random
import numpy as np

class Network(object):
    """a recurrent neural network with a bias at each step
    and using the sigmoid as the activation function
    """

    def __init__(self, wordsize):
        """initializes two weight matricies with Gaussian distributions and
        one bias vector to the zero vector

        wordsize -- the dimension of the input vectors to the network
        """
        # initializes two wordsize by wordsize weight arrays with Gaussian distributions N(0, 0.005) clipped to the range [-0.01, 0.01]
        self.hidden_weight, self.input_weight = tuple(np.clip(np.random.normal(0, 0.005, (wordsize, wordsize)), -0.01, 0.01) for i in range(2))

        # the bias is initialized to the 0 vector
        self.hidden_bias = np.zeros((wordsize, 1))

    def feedforward(self, x):
        """unfold the network and return two lists: one of pre-activation
        state vectors and one of post-activation state vectors

        x -- a list of vectors to be used as the input for the network
        """
        s = [None]*len(x)  # list to store the pre-activation state vectors
        h = [None]*len(x)  # list to store the post-activation state vectors

        s[0] = self.input_weight.dot(x[0]) + self.hidden_bias
        h[0] = sigmoid(s[0])

        for i in range(1, len(x)):
            s[i] = self.hidden_weight.dot(h[i-1]) + self.input_weight.dot(x[i]) + self.hidden_bias  # feed to next layer
            h[i] = sigmoid(s[i])  # activation function

        return s, h

    def backprop(self, x, s, h, nabla_start, truncate=100):
        """perform backpropagation through the network and return
        gradients with respect to the hidden_weight, input_weight, and
        hidden_bias, respectively

        x -- a list of vectors that were used as input to the network
        s -- the list of pre-activation state vectors produced by feedforward(x)
        h -- the list of post-activation state vectors produced by feedforward(x)
        nabla_start -- the gradient of the error function with respect to the final h, 
        to be used as a starting point for backpropagation
        truncate -- the number of layers to propagate backwards (default 100)
        """
        n = len(s)  # number of layers

        # arrays to store grad(C) wrt hidden_weight, input_weight, and hidden_bias, respectively
        nabla_U = np.zeros(self.hidden_weight.shape)
        nabla_W = np.zeros(self.input_weight.shape)
        nabla_b = np.zeros(self.hidden_bias.shape)

        # delta is the gradient of C with respect to the s vector of the current layer
        delta = np.multiply(nabla_start, sigmoid_prime(s[n-1]))  # start with the delta of the final layer

        # sum gradients either all the way to layer 0 or layer (n-truncate)
        for i in range(n-1, max(-1, n-truncate-1), -1): 
            if i > 0: nabla_U += np.outer(delta, h[i-1])  # add grad(C) wrt hidden_weight for layer i
            nabla_W += np.outer(delta, x[i])              # add grad(C) wrt input_weight for layer i
            nabla_b += delta                              # add grad(C) wrt hidden_bias for layer i
            delta = np.multiply((self.hidden_weight.T.dot(delta)), sigmoid_prime(s[i]))  # update delta for s[i-1]

        return nabla_U, nabla_W, nabla_b  # grad(C) wrt hidden_weight, input_weight, hidden_bias, respectively

class DoubleNetwork(object):
    """combination of two Network objects that takes the output of each network, 
    concatenates them, applies a final weight and bias, and transforms the result using softmax
    in order to output a distribution representative of the probability of the two 
    network inputs being the a given class
    """
    def __init__(self, wordsize, num_classes):
        """initializes two Network objects, a final weight to a Gaussian distribution,
        and a final bias to the zero vector

        wordsize -- the dimension of the input vectors to the networks
        num_classes -- the number of classes for the network to categorize into
        """
        self.net1 = Network(wordsize)
        self.net2 = Network(wordsize)

        # a final weight matrix that will transform the concatenation of the two network outputs into a vector with dimension equal to the number of classes
        # initialized to a Gaussian distribution N(0, 0.005) clipped to [-0.01, 0.01]
        self.final_weight = np.clip(np.random.normal(0, 0.005, (num_classes, 2*wordsize)), -0.01, 0.01)

        self.final_bias = np.zeros((num_classes, 1))

    def feedforward(self, x1, x2):
        """feeds x1 and x2 through the two Network objects, concatenates
        their inputs, applies a final weight and bias, transforms the
        output with softmax, and returns 6 lists: s1, s2, h1, h2, h, and p
        where s1 and s2 are the lists of pre-activation state vectors for each network,
        h1 and h2 are the lists of post-activation state vectors for each network,
        h is the concatenation of the final post-activation state vectors from each network,
        and p is the softmax-transformed probability distribution over the classes

        x1 -- list of input vectors for the first network
        x2 -- list of input vectors for the second network
        """
        s1, h1 = self.net1.feedforward(x1)
        s2, h2 = self.net2.feedforward(x2) 
        h = np.concatenate((h1[-1], h2[-1]))  # output of two networks combined

        p = softmax(np.dot(self.final_weight, h) + self.final_bias)  # final output probability distribution
        return s1, s2, h1, h2, h, p

    def backprop(self, x1, x2, s1, s2, h1, h2, h, p, target, truncate):
        """perform backpropagation through both networks and return
        gradients with respect to the hidden_weight of network 1,
        input_weight of network 1, hidden_bias of network 1, hidden_weight
        of network 2, input_weight of network 2, hidden_bias of network 2,
        final_weight, and final_bias, respectively

        x1 -- a list of vectors that were used as input to the first network
        x2 -- a list of vectors that were used as input to the second network
        s1, s2, h1, h2, h, p -- the output of feedforward(x1, x2)
        target -- a one-hot column vector corresponding to the correct class
        truncate -- the number of layers to propagate backwards in each network (default 100)
        """
        nabla_f = p - target  # grad(C) wrt final bias
        nabla_V = np.outer(nabla_f, h)  # grad(C) wrt final weight
        nabla_h = self.final_weight.T.dot(nabla_f)  # grad(C) wrt final hidden vector, used as starting point for backprop of each network

        nabla_U1, nabla_W1, nabla_b1 = self.net1.backprop(x1, s1, h1, np.split(nabla_h, 2)[0], truncate)
        nabla_U2, nabla_W2, nabla_b2 = self.net2.backprop(x2, s2, h2, np.split(nabla_h, 2)[1], truncate)

        return nabla_U1, nabla_W1, nabla_b1, nabla_U2, nabla_W2, nabla_b2, nabla_V, nabla_f

    def SGD(self, data, batch_size, eta, epochs, truncate, test_data=None):
        """trains the DoubleNetwork using stochastic gradient descent by dividing
        the data into batches of batch_size and running train_batch on each one, repeating
        for a certain number of epochs

        data -- a list of tuples in the form (x1, x2, target) representing all the training data
        batch_size -- the number of training data points in each batch for
        the gradients to be averaged across before updating the weights and biases
        eta -- the learning rate
        epochs -- the number of epochs to train for
        truncate -- the number of layers to propagate backwards in each network (default 100)
        test_data -- an optional tuple in the form (x1, x2) and if included,
        will be evaluated and the resulting probability distribution p will be
        printed after each epoch (default None)
        """

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
        """updates the weights and biases of the DoubleNetwork using
        a batch of training data by averaging the gradients over the batch
        and subtracting the averaged gradients

        batch -- a list of tuples in the form (x1, x2, target)
        eta -- the learning rate
        truncate -- the number of layers to propagate backwards in each network (default 100)
        """
        # variables to accumlate the gradients wrt each weight and bias
        nabla_U1 = nabla_W1 = np.zeros(self.net1.hidden_weight.shape) 
        nabla_U2 = nabla_W2 = np.zeros(self.net2.hidden_weight.shape)
        nabla_b1 = np.zeros(self.net1.hidden_bias.shape)
        nabla_b2 = np.zeros(self.net2.hidden_bias.shape)
        nabla_V = np.zeros(self.final_weight.shape)
        nabla_f = np.zeros(self.final_bias.shape)
        for x1, x2, target in batch:
            s1, s2, h1, h2, h, p = self.feedforward(x1, x2)
            # adds the results of backprop to each corresponding nabla_ variable
            nabla_U1, nabla_W1, nabla_b1, nabla_U2, nabla_W2, nabla_b2, nabla_V, nabla_f = tuple(sum(x) for x in zip(
                (nabla_U1, nabla_W1, nabla_b1, nabla_U2, nabla_W2, nabla_b2, nabla_V, nabla_f), 
                self.backprop(x1, x2, s1, s2, h1, h2, h, p, target, truncate)))

        self.net1.hidden_weight -= nabla_U1 * eta / len(batch)
        self.net1.input_weight -= nabla_W1 * eta / len(batch)
        self.net1.hidden_bias -= nabla_b1 * eta / len(batch)

        self.net2.hidden_weight -= nabla_U2 * eta / len(batch)
        self.net2.input_weight -= nabla_W2 * eta / len(batch)
        self.net2.hidden_bias -= nabla_b2 * eta / len(batch)

        self.final_weight -= nabla_V * eta / len(batch)
        self.final_bias -= nabla_f * eta / len(batch)

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
    return -1 * np.asscalar(np.log(p.tolist()[y.tolist().index([1])])) # -ln(p[j]) where j is the correct class
 
'''net = DoubleNetwork(10, 6)
sentence1 = [np.random.normal(0, 1, (10, 1)) for i in range(7)]
sentence2 = [np.random.normal(0, 1, (10, 1)) for i in range(4)]
targ = np.array([0, 0, 0, 1, 0, 0])
targ = np.reshape(targ, (6, 1))

s1, s2, h1, h2, h, p = net.feedforward(sentence1, sentence2)

U1, W1, b1, U2, W2, b2, V, f = net.backprop(sentence1, sentence2, s1, s2, h1, h2, h, p, targ, 100)
'''
