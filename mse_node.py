from base_node import Node
import numpy as np

class MSE(Node):
    def __init__(self, y, y_hat):
        """A node that computes the mean squared error.
        Should only be used at the last node in a network."""
        Node.__init__(self, [y, y_hat])

    def forward(self):
        """Compute the mean squared errror value for the node."""
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        y_hat = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.error = y - y_hat

        self.value = np.mean(np.square(self.error))

    def backward(self):
        """Compute the gradient for the MSE."""
        self.gradients[self.inbound_nodes[0]] = (2. / self.m) * self.error
        self.gradients[self.inbound_nodes[1]] = (-2. / self.m) * self.error
