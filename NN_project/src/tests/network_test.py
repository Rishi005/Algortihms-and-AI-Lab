import unittest
from architecture import Network, sigmoid, sigmoid_prime
import numpy as np
import mnist_loader
import random 

class TestNetwork(unittest.TestCase):
    def setUp(self):
        # not going to initialize anything here since the tests run faster when its initialized in the methods themselves e.g loading training data
        print("Set up goes here")

    def test_sigmoid_properties(self):
        """Test that the sigmod function works properly and produces the correctly shaped results within its boundry conditions (value must be between 0 and 1)"""
        z = np.array([-20, 0, 20])
        out = sigmoid(z)
        self.assertTrue(out.shape == z.shape)
        self.assertTrue(np.all(out > 0) and np.all(out < 1))
        self.assertTrue(np.isclose(sigmoid(0), 0.5))

    def test_sigmoid_prime(self):
        """Test that the derivative of the sigmoid function works properly"""
        self.assertTrue(np.isclose(sigmoid_prime(0), 0.25))
        z = np.linspace(-5, 5, 10)
        expected = sigmoid(z) * (1 - sigmoid(z))
        self.assertTrue(np.allclose(sigmoid_prime(z), expected))


    def test_network_init_shapes(self):
        """Testing if the shapes of the weights and biases are set up correctly during network initialization"""
        list_of_layers = [[784, 30, 10], [800, 50, 20, 10]]

        for layers in list_of_layers:
            net = Network(layers)
            self.assertTrue(len(net.weights) == len(layers)-1)

            for i in range(1, len(layers)):
                self.assertTrue(net.weights[i-1].shape == (layers[i], layers[i-1]))
                self.assertTrue(net.biases[i-1].shape == (layers[i], 1))

    def test_feedforward_shapes(self):
        """Test that the feed forward method produces activations and zs of the correct shape"""
        layers = [784, 30, 10]
        net = Network(layers)
        image = np.random.randn(layers[0], 1) #randomly initialize a flattened image with the size of the input layer
        activations, zs = net.feed_forward(image)
        self.assertTrue(len(activations) == len(layers))
        self.assertTrue(len(zs) == len(layers)-1) # excluding input layer
        self.assertTrue(activations[-1].shape == (layers[-1], 1))

    def test_backprop_shapes(self):
        """Test that the backpropagaion method returns the correct shapes and the gradients are not infinte"""
        layers = [784, 30, 10]
        net = Network(layers)
        image = np.random.randn(layers[0], 1)
        label = np.array([[1]])
        gb, gw = net.backprop(image, label) #use back propagation to get the gradients

        #check that the shapes returned by the backpropagation match and that the values are reasonable (finite)
        self.assertTrue(all(gb[i].shape == b.shape for i, b in enumerate(net.biases)))
        self.assertTrue(all(gw[i].shape == w.shape for i, w in enumerate(net.weights)))
        self.assertTrue(np.all(np.isfinite(np.concatenate([g.flatten() for g in gb]))))
        self.assertTrue(np.all(np.isfinite(np.concatenate([g.flatten() for g in gw]))))


    def test_zero_learning_rate(self):
        """Test that a learning rate of zero will not modify the weights"""
        np.random.seed(0)
        layers = [784, 30, 10]
        net = Network(layers)

        image = np.random.randn(layers[0], 1) 
        label = np.array([[1]])
        mini_batch = [(image, label)]

        old_weights = [w.copy() for w in net.weights]
        net.update_mini_batch(mini_batch, eta=0.0) #update the gradients with a learning rate of 0, which should mathematically not cause any changes in the weights
        for w, old_w in zip(net.weights, old_weights):
            self.assertTrue(np.allclose(w, old_w)) #check that they did not change

    def test_feed_forward_all_neurons(self):
        """Test that the feed forward method goes through all neurons in all layers"""
        layers = [784, 30, 10]
        net = Network(layers)
        a = np.random.randn(layers[0], 1) #initializing feedforward input by randomizing activaions for the input layer -- this in practice would be the greyscale pixel values of a flattened image
        activations, zs = net.feed_forward(a)
        
        num_neurons_a = sum([len(layer) for layer in activations[1:]]) # skipping the input layer (not real neurons)
        num_neurons_z = sum([len(layer) for layer in zs])

        self.assertEqual(num_neurons_a, sum(layers[1:]))
        self.assertEqual(num_neurons_z, sum(layers[1:])) # the zs are only computed for the hidden and output layers since the input layer isn't composed of actual neurons
    
    def test_nonzero_and_decreasing_gradients(self):
        """Test that none of the gradients are 0 and that at least some of them are decreasing in the backpropogation method"""
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        training_data = list(training_data)
        mini_batch = training_data[:10]
        layers = [784, 30, 10]
        net = Network(layers)

        for image, label in mini_batch:
            delta_gradient_b, delta_gradient_w = net.backprop(image, label)
            all_b = np.concatenate([b.flatten() for b in delta_gradient_b])
            all_w = np.concatenate([w.flatten() for w in delta_gradient_w])

            # check that not all are zero -- they aren't dead neurons and a small change in direction will result in a small change in the weights and biases 
            self.assertFalse(np.all(all_b == 0))
            self.assertFalse(np.all(all_w == 0))

            # check they contain some negative values (i.e., decreasing gradients to reach the minimum loss)
            self.assertTrue(np.any(all_b < 0))
            self.assertTrue(np.any(all_w < 0))


    def test_order_doesnt_affect_output(self):
        """Test that the order of input images does not affect the final weights and biases calculated in the update minibatch method"""
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        training_data = list(training_data)
        mini_batch = training_data[:10]

        np.random.seed(15) #initialize a seed so that the random initialization of the weights and biases once the Network object is created stays the same for every new object
        net = Network([784, 30, 10])
        net.update_mini_batch(mini_batch, eta=3.0)
        
        all_prev_b = np.concatenate([b.flatten() for b in net.biases])
        all_prev_w = np.concatenate([w.flatten() for w in net.weights]) #save weights and biases after training on an unshuffled minibatch to use for comparision later

        for trial in range(4):
            net = None # reset net object to none for a clean slate
            np.random.seed(15) #make sure the new object has the same initializations of weights and biases
            net = Network([784, 30, 10])
            random.shuffle(mini_batch) #shuffle the order of input images
            net.update_mini_batch(mini_batch, eta=3.0)
            all_curr_b = np.concatenate([b.flatten() for b in net.biases])
            all_curr_w = np.concatenate([w.flatten() for w in net.weights])

            self.assertTrue(np.allclose(all_prev_w, all_curr_w))
            self.assertTrue(np.allclose(all_prev_b, all_curr_b))

            all_prev_w = all_curr_w # update previous
            all_prev_b = all_curr_b


    def test_numerical_gradient_check(self):
        """
        Numerically estimate the gradient of the cost function w.r.t. weights and biases.
        Compares to analytical gradients from backpropagation following this stack overflow post:
        https://ai.stackexchange.com/questions/3962/how-do-i-know-if-my-backpropagation-is-implemented-correctly
        and this link
        https://web.archive.org/web/20171122205139/http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization 
        """
        np.random.seed(0)
        net = Network([784, 30, 10])
        x = np.random.randn(784, 1)
        y = np.random.randn(1, 1)


        analytic_b, analytic_w = net.backprop(x, y) #only computed the gradients, does not change the random initialization of the weights and biases


        num_w = [np.zeros_like(w) for w in net.weights]
        num_b = [np.zeros_like(b) for b in net.biases]


        # numerical gradients for weights
        #iterate through every weight individually
        epsilon=1e-5
        for l, W in enumerate(net.weights):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W_epsilon = np.copy(W)
                    W_epsilon[i, j] += epsilon
                    net.weights[l] = W_epsilon #reassigning the weight in the network so we can use the feedforward method to get the corresponding activations
                    plus_cost = 0.5 * np.sum((net.feed_forward(x)[0][-1] - y)**2) #This is the sum of our cost function: 0.5*(a - y)^2, the derivative will just be (a - y)


                    W_epsilon[i, j] -= 2 * epsilon
                    net.weights[l] = W_epsilon
                    minus_cost = 0.5 * np.sum((net.feed_forward(x)[0][-1] - y)**2) 


                    num_w[l][i, j] = (plus_cost - minus_cost) / (2 * epsilon) #compute derivative for one individual weight
                net.weights[l] = np.copy(W)  # reset layer


        # numerical gradients for biases
        for l, b in enumerate(net.biases):
            for i in range(b.shape[0]):
                b_epsilon = np.copy(b)
                b_epsilon[i, 0] += epsilon
                net.biases[l] = b_epsilon
                plus_cost = 0.5 * np.sum((net.feed_forward(x)[0][-1] - y)**2)


                b_epsilon[i, 0] -= 2 * epsilon
                net.biases[l] = b_epsilon
                minus_cost = 0.5 * np.sum((net.feed_forward(x)[0][-1] - y)**2)


                num_b[l][i, 0] = (plus_cost - minus_cost) / (2 * epsilon) 
            net.biases[l] = np.copy(b)  

        for ab, nb in zip(analytic_b, num_b):
            self.assertTrue(np.allclose(ab, nb))
        for aw, nw in zip(analytic_w, num_w):
            self.assertTrue(np.allclose(aw, nw))

