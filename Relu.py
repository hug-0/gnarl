from Node import Node
import numpy as np

class ReLU(Node):
    def __init__(self, node, epsilon=1e-4):
        """Computes rectified linear units for the node."""
        Node.__init__(self, [node])
        self.epsilon = epsilon

    def forward(self):
        """Forward propagate node values."""
        eps = np.zeros_like(self.inbound_nodes[0].value) + self.epsilon
        self.value = np.maximum(eps, self.inbound_nodes[0].value)

    def backward(self):
        """Backward propagate node gradients"""
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            grad_cost[self.value <= self.epsilon] = 0. # Kill gradients where value is 0.
            self.gradients[self.inbound_nodes[0]] += grad_cost
