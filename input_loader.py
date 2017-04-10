import nltk
import numpy as np
import network

class Loader(object):

    def __init__(self):
        self.wordvecs = {}
        self.scores = []  # one-hot vectors of similarity ratings from 0 to 5
        # list of lists of word vectors representing list of input sentences
        self.sentences1 = [] 
        self.sentences2 = []
    
    def load_word_vectors(self):
        """read in word vector data"""
        self.wordvecs = {} 
        with open("glove\\glove.6B.100d.txt", "r", encoding="UTF-8") as f:
            for line in f:
                l = line.split(' ')
                self.wordvecs[l[0]] = [float(i) for i in l[1:]]

    def load_training_data(self):
        """read in training data"""
        s1 = [] # first sentences
        s2 = [] # second sentences
        
        with open("sts2015-en-post\\data\\clean\\text.clean", "r", encoding="UTF-8") as f:
            for line in f:
                l = line.split('\t')
        
                # convert category to one-hot vector
                s = [0, 0, 0, 0, 0, 0]
                s[round(float(l[0]))] = 1
                self.scores.append(np.reshape(np.array(s), (6, 1)))
        
                s1.append(l[4])
                s2.append(l[5])

        # map words to word vectors and convert into format usable by neural network
        self.sentences1 = [[]] * len(self.scores) # list of lists of vectors
        self.sentences2 = [[]] * len(self.scores)
        
        for i in range(len(self.scores)):
            self.sentences1[i] = self.sentence_to_veclist(s1[i])
            self.sentences2[i] = self.sentence_to_veclist(s2[i])

    def sentence_to_veclist(self, s):
        """convert input sentence (as a string) to a list of vectors for the neural network to operate on"""
        words = nltk.word_tokenize(s)
        return [np.reshape(np.array(self.wordvecs.get(words[i])), (100, 1)) for i in range(len(words)) if words[i] in self.wordvecs.keys()]


# SGD(data, batch_size, eta, epochs, truncate)

loader = Loader()
loader.load_word_vectors()
loader.load_training_data()
net = network.DoubleNetwork(100, 6)

test_x1 = loader.sentence_to_veclist("it's pretty difficult to imagine a person with social anxiety disorder being an extrovert.")
test_x2 = loader.sentence_to_veclist("on the surface, it does seem like social anxiety disorder and extroversion shouldn't both exist in the same person.")

print("starting SGD")
net.SGD(list(zip(loader.sentences1, loader.sentences2, loader.scores)), 30, 0.01, 30, 100, (test_x1, test_x2))
