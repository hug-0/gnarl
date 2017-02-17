from base_node import Node
import numpy as np

class LeakyReLU(Node):
    def __init__(self, node, epsilon=0., leak=1e-2):
        """Computes leaky rectified linear units for the node."""
        Node.__init__(self, [node])
        self.epsilon = epsilon
        self.leak = leak

    def forward(self):
        """Forward propagate node values."""
        #eps = np.zeros_like(self.inbound_nodes[0].value) + self.epsilon

        #print('Forward bef:', self.inbound_nodes[0].value)
        self.value = np.maximum(self.epsilon, self.inbound_nodes[0].value)
        #print('Forward aft:', self.value)

    def backward(self):
        """Backward propagate node gradients"""
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            grad_cost[self.value <= self.epsilon] = self.leak

            self.gradients[self.inbound_nodes[0]] += grad_cost
