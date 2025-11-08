import numpy as np

# print(np.__version__)

def sigmoid(z):
    ''' The sigmoid activation function
    :param z: The input value
    :return: The output of the sigmoid function applied to z
    '''
    return 1.0 / (1.0 + np.exp(-z))

class Network:
    ''' Initializes a neural network with the given number of neurons per layer
    :param sizes: A list of the number of neurons in that given layer, e.g: [2, 3, 2] means there are 3 layers where the second layer has 3 neurons etc.
    '''

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # in the following we ignore the first layer since we assume that it's the input layer (not actually neurons, just data)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #list of matricies of randomly initialized biases for the neurons in each layer in the network using a standard gaussian distribution
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # list of matricies of weights between all neurons orgalized by layer

    def feed_forward(self, a):
        ''' Feedforward the input a through the network
        :param a: The input to the network
        :return: The output of the network after feedforwarding x
        '''
        for b, w in zip(self.biases, self.weights): 
            a = sigmoid(np.dot(w, a) + b) # vectorized for each layer
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        pass

# net = Network([2, 3, 2])
# print("Biases:", net.biases)
# print("Weights:", net.weights)