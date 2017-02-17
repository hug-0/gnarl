from base_node import Node
import numpy as np

class Softmax(Node):
    def __init__(self, node):
        """A node that represents the softmax activation function.
        Should always be last node in a graph before computing loss, and only useful for probabilistic
        outputs between 0 and 1."""
        Node.__init__(self, [node])

    def _softmax(self, z):
        """Compute the softmax probability for the inputs."""
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self):
        """Forward propagate node value, e.g. probabilities."""
        self.value = self._softmax(self.inbound_nodes[0].value)

    def backward(self):
        """Backward propagate gradient weights."""
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # NOT CORRECT. TODO.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            grad_cost[self.value]

            self.gradients[self.inbound_nodes[0]] += grad_cost
