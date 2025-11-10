import numpy as np
import random

# print(np.__version__)

def sigmoid(z):
    ''' The sigmoid activation function
    :param z: The input value
    :return: The output of the sigmoid function applied to z
    '''
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    ''' Derivative of the sigmoid function
    :param z: The input value
    :return: The derivative of the sigmoid function applied to z
    '''
    return sigmoid(z) * (1 - sigmoid(z))

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
    
    def backprop(self, image, label):
        pass
    
    def update_mini_batch(self, mini_batch, eta):
        """
        Updates the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        Applies the update rule onto the weights and biases for the Network object using one mini batch of training samples.
        
        :param mini_batch: A list of tuples (image, label) representing the mini batch
        :param eta: The learning rate
        :return: None
        """
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # Increment the gradient of the weights and biases by the small change from going by a small step in the direction of the minimum of loss function for every sample in the minibatch (Xi)
        # Let the samples from mini batch = X1, ..., Xm, where any one sample can be denoted by Xi. 
        # The below is equivalent to sum(gradient_loss_funct_Xi)
        for image, label in mini_batch:
            #using the chain rule to compute the gradient of the loss function with respect to weights and biases for a single sample in the minibatch
            delta_gradient_b, delta_gradient_w = self.backprop(image, label)

            gradient_b = [gb+dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw+dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

        # Apply the update rule on the weights and biases using the estimation from a minibatch
        # Then the estimation is: gradient_loss_funct ≈ sum(gradient_loss_funct_all_samples) / n ≈ sum(gradient_loss_funct_Xi) / m
        # Essentially stating that the gradient of the loss function can be approximated by the average of gradients over a minibatch of samples
        # Therefore in the update rule the gradient of the loss function is replaced by this approximation:
        self.weights = [w - (eta/len(mini_batch))*gw for w, gw in zip(self.weights, gradient_w)]
        self.biases = [b - (eta/len(mini_batch))*gb for b, gb in zip(self.biases, gradient_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 
           
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"epoch {j} complete.")

# net = Network([2, 3, 2])
# print("Biases:", net.biases)
# print("Weights:", net.weights)