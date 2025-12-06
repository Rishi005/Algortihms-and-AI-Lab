# How the Network works

## 1. Input vector x
Set an activation vector $a^1$ for the first layer as the pixel values of the image.

Because here we use for any subsequent layer l 

$a^l = \sigma(z^l) = \sigma(\sum w^la^{l-1} + b^l)$

where $\sigma(z)$ is the sigmoid function $\dfrac{1}{1+e^{-z}}$


## 2. Feed forward
Feed the activation forward throughout the network and compute $a^l$ for each 
layer until the last layer $L$

## 3. Compute output error
Compute error for the output layer $L$ by using the following equation derived using chain rule:

$\delta^L = \nabla_z C = \nabla_a C \odot \sigma'(z^L)$


Where $\odot$ is the elementwise product (Hadamard's product) between the gradient of the cost/loss function wrt. the activation and the derivative of the activation of that layer

> Proof: \
$\delta^l = \nabla_z C = \dfrac{\partial C}{\partial a^l} \dfrac{\partial a^l}{\partial z^l} $
\
\
We also know $a^l = \sigma(z^l)$
\
$\therefore \dfrac{\partial a^l}{\partial z^l} = \dfrac{\partial \sigma(z^l)}{\partial z^l} = \sigma'(z^l)$
\
\
Hence we can simplify as
\
$\delta^l = \dfrac{\partial C}{\partial a^l}  \sigma'(z^l) = \nabla_a C \odot \sigma'(z^l)$

This essentially tells us how fast the cost function changes wrt. each neuron in that layer and how fast the activation function changes at the $L$ th layer

We can simplify this: since we know are using the Mean Squared Error (MSE) as our cost function, we can compute the gradient of the MSE wrt to $a^l$

$MSE = \dfrac{1}{2} || y + a^l||_2^2$ &emsp; where y is the true label

$\therefore \dfrac {\partial MSE}{\partial a^l} = \dfrac{2}{2} || y + a^l||_2 \cdot (0-1)$ 

&emsp; &emsp; &emsp;&emsp;&emsp;$= a^l - y$

Hence $\delta^L = (a^L - y) \odot \sigma'(z^L)$ &emsp;&emsp; **(Equation 1)**

## 4. Backpropagate the Error

Here we start using the confusing but useful notation where layer $l+1$ is our current layer and layer $l$ is the previous layer. WE calculate the errors fo5r all the previous layers 

$l = L-1, ..., 2$

We can denote the error of the previous layer as 

$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$ &emsp;&emsp; **(Equation 2)**

Which we can again derive using chain rule on $\delta^l = \nabla_z C$

> Proof: \
We cannot work with the vectorized form here, so let $j$ denote the j'th neuron in layer $l$ and $k$ denote the k'th neuron in layer $l+1$
\
We want to rewrite $\delta_j^l$ in terms of $\delta_k^{l+1}$. Using the chain rule again we get:
\
\
$\delta_j^l = \dfrac{\partial C}{\partial z_j^l}$
\
\
$ = \sum_k \dfrac{\partial C}{\partial z_k^{l+1}} \dfrac{\partial z_k^{l+1}}{\partial z_j^l}$
\
\
$ = \sum_k \delta_k^{l+1}\dfrac{\partial z_k^{l+1}}{\partial z_j^l}$
\
\
Note that $z_j^l = \sum_h w_{jh}^l a_h^{l-1} + b_j^l $ where $h$ is the h'th neuron in layer $l-1$
\
\
Hence $z_k^{l+1} = \sum_j w_{kj}^{l+1} a_j^{l} + b_k^{l+1} $
\
\
$= \sum_j w_{kj}^{l+1} \sigma(z_j^{l}) + b_k^{l+1} $ 
\
\
$\therefore \dfrac{\partial z_k^{l+1}}{\partial z_j^l} = \dfrac{\partial (\sum_j w_{kj}^{l+1} \sigma(z_j^{l}) + b_k^{l+1})}{\partial z_j^l}$
\
\
$= w_{kj}^{l+1} \sigma'(z_j^l) $
\
\
Subbing it in and putting everything together, we finally arrive at
\
$\delta_j^l = \sum_k (\delta_k^{l+1})(w_{kj}^{l+1} \sigma'(z_j^l)) $
\
\
Vectorized this is rewritten as
\
$\delta_j^l =(w^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$

We can use *Equation 1* to compute the error of our current layer $\delta^{l+1} $

## 5. SGD with components of $\nabla C$
Now we can get both $\nabla_w C$ and $\nabla_b C$: 

$\dfrac{\partial C}{\partial w^l} = a^{l-1} \delta^l$ 

$\dfrac{\partial C}{\partial b^l} = \delta^l$

We will use these two equations as our **update rule** during SGD

> Proof: \
**Part 1:** For $\dfrac{\partial C}{\partial w^l} = a^{l-1} \delta^l$ :
\
\
$\dfrac{\partial C}{\partial w^l} = \dfrac{\partial C}{\partial a^l} \dfrac{\partial a^l}{\partial z^l} \dfrac{\partial z^l}{\partial w^l} = (\delta^l) \dfrac{\partial z^l}{\partial w^l} $ 
\
\
Then 
\
$\dfrac{\partial z^l}{\partial w^l} = \dfrac{\partial (w^l a^{l-1} + b^l)}{\partial w^l} =  a^{l-1}$
\
\
$\therefore \dfrac{\partial C}{\partial w^l} = a^{l-1} \delta^l $
\
\
\
Similarly
\
**Part 2:** For $\dfrac{\partial C}{\partial b^l} = \delta^l$
\
\
$\dfrac{\partial C}{\partial b^l} = \dfrac{\partial C}{\partial a^l} \dfrac{\partial a^l}{\partial z^l} \dfrac{\partial z^l}{\partial b^l} $ 
\
\
We know $a^l = \sigma(z^l)$
\
And we also know $\dfrac{\partial z^l}{\partial b^l} = \dfrac{\partial (w^l a^{l-1} + b^l)}{\partial b^l} =  1$
\
\
$\therefore \dfrac{\partial C}{\partial b^l} = \dfrac{\partial C}{\partial a^l} \dfrac{\partial \sigma(z^l)}{\partial z^l} (1)$
\
\
$= \nabla_a C \odot \sigma'(z^l) = \delta^l $


