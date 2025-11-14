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
        activations = [a]
        zs = []
        for b, w in zip(self.biases, self.weights): 
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z) # vectorized for each layer
            activations.append(a)
        
        return activations, zs
    
    def cost_derivative(self, output_activations, label):
        """
        Returns the gradient of the loss function (Mean Squared Error in this case) with respect the the output activation a
        """
        return (output_activations - label)

    
    def backprop(self, image, label):
        """
        Back propagates from the output layer to the input layer, calculating and storing the components of the gradient of the loss function, that is 
        :math: \partial C / \partial w and \partial C / \partial b 
        for every layer for one image and label
        """

        # initialize the gradients as zero vectors
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # get the activation values and weighted inputs for every neuron in each layer through the feed forward algorithm
        activations, zs = self.feed_forward(image)

        # calculate error from output layer (using last activations in the list i.e. output layer activation and output layer weighted inputs zs)
        delta = self.cost_derivative(activations[-1], label) * sigmoid_prime(zs[-1]) 
        
        # subsitute the components of the gradient of the loss function with the components in terms of the error delta derived from the equations from back propagation
        # assign it to the last layer as we only have information about the output layer after the feed forward
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].T) # indexed second last since we want the activations from layer L-1, since we are going backwards where L = the last layer (index -1), hence L-1 is the second last layer
    
        # backpropagate the error to get  the small change in gradient_b, gradient_w for all layers before the output
        # iterate through the layers of the network backwards
        for l in range(-2, -self.num_layers, -1):
            z = zs[l]
            # use equation 2 to calculate the error in the previous layer
            # here layer l+1 represents our current layer and l is the previous layer
            # in this loop we are moving backwards in terms of indices: -2, -3, -4, ..., hence adding one (l+1) moves one index forward, i.e. from the previous index to our current index 
            # (we start with our current index at the output layer l=-1) 
            delta = np.dot(self.weights[l+1].T, delta) * sigmoid_prime(z)
            # assign the gradients of weights and biases of previous layer using the earlier substitution
            gradient_b[l] = delta
            gradient_w[l] = np.dot(delta, activations[l-1].T)

        return gradient_b, gradient_w

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. 
        
        The neural network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feed_forward(x)[0][-1]), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


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
            test_data = list(test_data)
            n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)

        for j in range(epochs):
            # partition the training data into randomly chosen minibatches in every epoch
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 
           
           # update the network for every mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                # evaluate how well the model performs against unseen test data in every epoch --> prints how many test samples were correctly predicted out of the total
                print(f"epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"epoch {j} complete.")



