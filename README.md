# **MoE Neural Network Prototype**

## Mixture of Experts (MoE) Formulation

The final prediction $y_t$ at time step $t$ is calculated as a weighted sum of the $N$ expert predictions:

$$y_t=\sum_{i=1}^{N}w_{t,i}e_{t,i}$$

Where:
* $e_{t,i}$ is the prediction of expert $i$ at time $t$.
* $w_{t,i}$ is the weight assigned to expert $i$ at time $t$, determined by the gating network.

---

## Gating Network Architecture

The weights are generated dynamically based on the input features using a shallow neural network. The process involves a hidden layer with a hyperbolic tangent activation, followed by a linear output layer that is normalized using a softmax function:

$$h_t=\tanh(W_{in}x_t+b_{in})$$

$$l_t=W_{out}h_t+b_{out}$$

$$w_t=\text{softmax}(l_t)$$

Where $x_t$ represents the input features (or a single specific feature) fed into the gating network at time $t$.

---

## Loss Function

The network is trained to minimize the Mean Squared Error (MSE) between the true target values and the aggregated predictions:

$$\text{MSE}=\frac{1}{T}\sum_{t=1}^{T}(y_t-\hat{y}_t)^2$$