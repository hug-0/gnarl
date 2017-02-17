# Import dependencies
import numpy as np
import sys, os, pickle
from sklearn.utils import shuffle, resample

# Import nodes
from base_node import Node
from input_node import Input
from linear_node import Linear

# Import loss nodes
from mse_node import MSE
from cross_entropy_node import CrossEntropy

# Import activation nodes
from sigmoid_node import Sigmoid
from softmax_node import Softmax
from leaky_relu_node import LeakyReLU
from relu_node import ReLU

# Activation functions
ACTIVATIONS = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax
}

# Loss functions
LOSSES = {
    'mse': MSE,
    'cross_entropy': CrossEntropy
}

class Gnarl(object):
    def __init__(self, X=None, y=None, activation='leaky_relu',
                 learning_rate=1e-4,
                 regularization=0.,
                 verbose=False,
                 loss='mse',
                 batch_size=10):
        """An instance of a neural net model.

        Creates a neural network model object.
        """

        # Init model from options
        self._init(X=X, y=y, activation=activation,
                   learning_rate=learning_rate, regularization=0., verbose=verbose, loss=loss, batch_size=batch_size)

    def _init(self, X=None, y=None, activation='leaky_relu',
              learning_rate=1e-4,
              regularization=0.,
              verbose=False,
              loss='mse',
              batch_size=10):
        """Initialize the model."""

        # Todo: ensure that options are correct

        # Init dicts to hold the layers and the weights and biases
        #self.layers = {}
        self.nodes = {}
        self.layers_list = []
        self._weights = []
        self._biases = []
        self.trainables = []

        # Init graph
        self.graph = []

        try:
            self.activation = activation
            self.learning_rate = learning_rate
            self.regularization = regularization
            self.verbose = verbose
            self.loss = loss
            self.batch_size = batch_size
        except ValueError as e:
            print('Init failed with error: ', e)

        # Set X and y input nodes to initiate model
        self.X = Input()
        self.y = Input()

        self._update_input(X)
        self._update_output(y)

        # Set first nodes in node dict for future
        self.nodes[self.X] = self.X.value
        self.nodes[self.y] = self.y.value

        self._input_layer(self.X.value)

    def _update_input(self, X_train):
        """Update the model's feature training data."""
        self.X.value = X_train

    def _update_output(self, y_train):
        """Update the model's output training data."""
        self.y.value = y_train

    def _input_layer(self, X_train):
        """Define the model's input layer."""
        self.layers_list.append(self.X)
        self._weights.append(self.nodes[self.X]) # Convenience for add_layer

    def _reset_graph(self):
        """Reset graph node values."""
        for node in self.trainables:
                # If weights
                if len(node.value.shape) == 2:
                    node.value = np.random.randn(node.value.shape[0], node.value.shape[1])
                # if bias
                elif len(node.value.shape) == 1:
                    node.value = np.zeros_like(node.value)

    def _log_probs(self):
        """Compute and return the log probabilities for the outputs of a classification problem."""
        output = self.graph[-2].value
        norm_probs = np.exp(output)/np.sum(np.exp(output), axis=1, keepdims=True)

        return np.log(norm_probs)

    def _probs(self):
        """Compute and return the normalized probabilities for the outputs of a classification problem."""
        output = self.graph[-2].value
        norm_probs = np.exp(output)/np.sum(np.exp(output), axis=1, keepdims=True)

        return norm_probs

    def add_layer(self, out_nodes,
                     activation):
        """Add a hidden layer to the model."""
        # Init random weights
        W = Input()
        W_ = np.random.randn(self._weights[-1].shape[1], out_nodes)
        self._weights.append(W_) # Store weights for convenience

        # Init biases
        b = Input()
        b_ = np.zeros(out_nodes)
        self._biases.append(b_) # Store biases for convenience

        # Add weights and biases to dicts for future use in connect_layers
        self.nodes[W] = W_
        self.nodes[b]= b_

        # Linear combo from previous layer to current layer
        layer = Linear(self.layers_list[-1], W, b)

        # Activation
        if activation == 'none': # Output layer is regression
            self.layers_list.append(layer)
        #elif activation == 'cross_entropy':
        #    self.layers_list.append(layer)
        elif activation == 'softmax':
             self.layers_list() # Should fix so that softmax is its own layer
             # Right now, the computation is done in the CrossEntropy node
        else:
            activation = ACTIVATIONS[activation](layer)
            self.layers_list.append(activation)

        # Add weights and biases to trainables
        self.trainables += [W, b]

    def connect_layers(self):
        """Connect and build the computational graph the network represents."""

        # Attach loss/cost function to output layer.
        if self.loss not in LOSSES:
            raise NotImplementedError

        if self.loss == 'cross_entropy':
            loss = CrossEntropy(self.y, self.layers_list[-1], reg=self.regularization)
        elif self.loss == 'mse':
            loss = MSE(self.y, self.layers_list[-1])

        # Create graph
        self.graph = _topological_sort(self.nodes)

    def fit(self, X_train, y_train, solver='sgd', epochs=10, fit_more_data=False):
        """Train the model."""

        # Reset graph values to purge nodes
        if fit_more_data is False:
            self._reset_graph()

        # Setup convenient vars
        m = X_train.shape[0]

        # Base case num runs
        steps_per_epoch = m // self.batch_size

        if self.verbose:
            print('Training model...')
            print('Solver:', solver)
            print('Total number of samples:', m)
            if solver == 'sgd':
                print('Steps per epoch:', steps_per_epoch)
            print('='*80)

        for i in range(epochs):
            loss = 0
            for j in range(steps_per_epoch):
                # Randomly sample a batch of examples if SGD
                if solver == 'sgd':
                    X_batch, y_batch = resample(X_train, y_train, n_samples=self.batch_size)

                # Update value of X and y Inputs
                self._update_input(X_batch)
                self._update_output(y_batch)

                # Forward and backward propagate graph
                _forward_and_backward(self.graph)

                # Update weights and biases
                _sgd_update(self.trainables, learning_rate=self.learning_rate)

                # Update loss
                if solver == 'sgd':
                    loss += self.graph[-1].value

            if self.verbose:
                if solver == 'sgd':
                    sys.stdout.write("\rEpoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
                    if i % 500 == 0:
                        print('') # Force new epoch printline
        if self.verbose:
            print('')
            print('='*80)
            print('Finished training model.')
            print('Final loss: %.3f' % self.graph[-1].value)

    def predict(self, X_test, truncate_labels=True):
        """Predict output using the trained model.

        Return predictions.
        """
        self._update_input(X_test)

        # Forward propagate (avoid last node, as this is the loss)
        for node in self.graph[:-1]:
            node.forward()

        # If regression / single output node
        if self.loss == 'mse':
            return self.graph[-2].value

        # If classification
        if self.loss == 'cross_entropy':
            # Turn predictions into probabilities
            norm_probs = self._probs()
            if not truncate_labels:
                # Return predictions as a matrix of probabilities
                return norm_probs
            else:
                # Return predictions as a one-dim vector with elements representing the class with the highest probability
                return np.argmax(norm_probs, axis=1)

# Utility functions to help with implementations and checks of Node classes
def _topological_sort(feed_dict):
    """
    Source: Udacity, 2016.

    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is an `Input` node and
                the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def _forward_and_backward(graph):
    """Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()

def _sgd_update(trainables, learning_rate=1e-4):
    """Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    for t in trainables:
        partial = t.gradients[t]
        t.value -= learning_rate * partial

def save_model(model, model_name, path='./'):
    """Save a Gnarl model as a pickle file."""
    assert type(model_name) == str
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if os.path.exists(path):
        with open(os.path.join(path, model_name + '.pickle'), 'wb') as f:
            try:
                pickle.dump(model, f, protocol=-1)
            except Exception as e:
                print("Couldn't save Gnarl model %s to file %s." % model_name, path)

def load_model(file_path):
    """Load a Gnarl model from a pickle file."""
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception as e:
                print("Couldn't load Gnarl model from file %s" % file_path)
