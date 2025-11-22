Basic unit tests and a larger test to check the backpropagation works as intended were created:

test_sigmoid_properties() → verifies that the input shape is preserved, that all outputs lie strictly between 0 and 1, and that the function returns the analytically correct value for a known input (sigmoid(0) = 0.5)

test_sigmoid_prime() → verifies correctness of the derivative of the sigmoid by comparing it against the analytically expected result σ(z)(1−σ(z)) for both a scalar and a vector of inputs

test_network_init_shapes() → verifies that a newly created network initializes the correct number of weight and bias matrices and that each matrix has the correct shape according to the layer sizes

test_feedforward_shapes() → verifies that feed-forward returns the correct number of activation vectors and z-vectors and ensures the output activation has the correct shape for the final layer

test_backprop_shapes() → verifies that the gradients returned by backprop match the shapes of the network’s weights and biases and that all gradient values are finite (no NaNs or infinities)

test_zero_learning_rate() → verifies that using a learning rate of 0 during a minibatch update leaves all weight matrices unchanged compared to their original values

test_feed_forward_all_neurons() → verifies that feed-forward processes every neuron in all hidden and output layers by checking that total neuron counts in activations and z-values match the architecture

test_nonzero_and_decreasing_gradients() → verifies that backpropagation produces gradients that are not all zero (no dead layers) and contain some negative values, indicating proper direction toward loss minimization

test_order_doesnt_affect_output() → verifies that shuffling the order of samples in a minibatch does not change the resulting weight and bias updates when starting from identical initializations

test_numerical_gradient_check() → verifies the correctness of backpropagation by comparing analytically computed gradients to numerically estimated gradients using finite differences for every weight and bias

Realistic inputs were used, mostly the same size as the current model takes, so a network shape of [784, 30, 10]. 
To reproduce these tests, you need to install the following dependencies using poetry: unittests, numpy. You will also need to download the mnist_loader.py file and architecture.py file since the tests will be using methods from these. The tests (in network.tests.py) need to be inside the src folder, which also contains mnist_loader.py and architecture.py. To run these tests, you can navigate into your src folder in the terminal and type in the following command: pytest network_test.py.  


The coverage test is as follows:

<img width="975" height="237" alt="Screenshot from 2025-11-22 14-51-25" src="https://github.com/user-attachments/assets/af0c5a3f-933d-4da3-a3c5-1c4f9645e48c" />
