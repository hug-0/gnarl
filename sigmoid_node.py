from base_node import Node
import numpy as np

class Sigmoid(Node):
    def __init__(self, node):
        """Compute the sigmoid of a given input node."""
        Node.__init__(self, [node])

    def _sigmoid(self, z):
        """Compute the sigmoid for an input z."""
        return 1./(1 + np.exp(-z))

    def forward(self):
        """Compute the value of the Sigmoid node."""
        Z = self.inbound_nodes[0].value
        self.value = self._sigmoid(Z)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            dZ = sigmoid * (1. - sigmoid)
            self.gradients[self.inbound_nodes[0]] += dZ * grad_cost # Ele wise
