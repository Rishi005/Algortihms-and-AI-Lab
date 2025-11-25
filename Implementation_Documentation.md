The general structure of this program is that the neural network learns on training data with stochastic gradient descent, where each mini-batch of data is updated using the gradients computed from the backpropagation algorithm. The backpropagation gets the current activations of the network through the feed forward method, and then calculates and stores the gradient of the cost function with respect to the weights and biases with the current activations and the error (from the derivative of the cost function), moving backwards in the network. 

## The Big-O analysis for training one minibatch:

In our case, there are 3 layers, let's denote $n_1$, $n_2$, and $n_3$ as the number of neurons in each layer, where $n_1$ are in the input layer, $n_2$ is the hidden layer, and $n_3$ is the output layer.  Hence the number of weights overall for one training data point is $(n_1 \cdot n_2) + (n_2 \cdot n_3)$, which we can call the weight matrices $W_1 + W_2$. The time complexity is determined by the number of matrix multiplications needed, since these are the longest operations in the algorithm. Let each mini-batch have $d$ training data points, then the number of operations for the matrix multiplications becomes $O(W_1 \cdot d) = O(d \cdot n_1 \cdot n_2)$, since we are computing a weight matrix of the size $W_1$, $d$ times for each training data point. The time complexity for the forward pass from the hidden layer to the output layer can be calculated the same way: we get $O(W_2 \cdot d) = O(d \cdot n_2 \cdot n_3)$. The backpropagation’s time complexity is the same as the forward passes, since we are computing the gradients with respect to the weights for each layer and each training data point, so for matrices of size $W_2$ and $W_1$, d times. Then for the last step where we update every minibatch’s weights and biases in SGD, the time complexity is $O(W_1)$ and $O(W_2)$, which are $O(n_1 \cdot n_2) + O(n_2 \cdot n_3)$ respectively. 

Hence, for the forward pass we get a total time complexity of $O(d \cdot n_1 \cdot n_2) + O(d \cdot n_2 \cdot n_3)$, and for the backwards pass a total time complexity of $O(d \cdot n_2 \cdot n_3) + O(d \cdot n_1 \cdot n_2)$, and for the gradient updates a total time complexity of $O(n_1 \cdot n_2) + O(n_2 \cdot n_3)$. 

Putting these all together we get $O(d \cdot n_1 \cdot n_2) + O(d \cdot n_2 \cdot n_3) + O(d \cdot n_1 \cdot n_2) + O(d \cdot n_2 \cdot n_3) + O(n_1 \cdot n_2) + O(n_2 \cdot n_3)$

$= O(d \cdot n_1 \cdot n_2 + d \cdot n_2 \cdot n_3 + d \cdot n_1 \cdot n_2 + d \cdot n_2 \cdot n_3 + n_1 \cdot n_2 + n_2 \cdot n_3)$

$= O(d \cdot n_2 \cdot (n_1 + n_3) + n_2\cdot(n_1 + n_3))$

Then for space complexity, we need to calculate how much memory is needed to store all the calculations done. Firstly, we need to store our training data minibatch, which has $n_1$ pixels each, hence the space complexity of that comes to $O(n_1 \cdot d)$. We also need to store the activations from the hidden and output layers, we have one activation per neuron for each training data point, hence we get $O(n_2 \cdot d)$ and $O(n_3 \cdot d)$ respectively. Then we also need to store all of the weights, hence we get a space complexity of $O(W_1) + O(W_2) = O(n_1 \cdot n_2) + O(n_2 \cdot n_3)$. Then finally we need to also store the computed gradients, which is the same as the complexity for the weights, so $O(n_1 \cdot n_2) + O(n_2 \cdot n_3)$. 

In total we get a space complexity of: 

$O(n_1 \cdot d) + O(n_2 \cdot d) + O(n_3 \cdot d) + O(n_1 \cdot n_2) + O(n_2 \cdot n_3)$. 

$= O(d \cdot (n_1 + n_2 + n_3) + n_2 \cdot (n_1 + n_3))$

The space and time complexities of the biases are dominated by those of the weights and gradients, hence is insignificant.
***
## Sources and use of large language models in the project

No AI was used for writing the code or tests. ChatGPT was used to help understand the time and space complexities for this document. 

The following sources were used in this project:

For understanding the math and the code: [Michael A. Nielsen's, "Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/chap1.html)

For writing more comprehensive tests: [Sebastian Bjorkqvist: Writing automated tests for neural networks](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)

For Big-O analysis: 

https://ai.stackexchange.com/questions/13612/what-is-the-time-complexity-of-the-forward-pass-algorithm-of-a-feedforward-neura 

https://www.quora.com/What-is-the-time-complexity-of-the-forward-pass-algorithm-of-a-feedforward-neural-network 
