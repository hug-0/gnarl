from base_node import Node
import numpy as np

class Linear(Node):
    def __init__(self, *args):
        """A node that computes the linear combination of a list of input
        nodes features, a list of input weights, and a bias term."""
        Node.__init__(self, [*args])

    def forward(self):
        """Compute the linear combination of the inbound nodes."""

        # Create vectors
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value

        if self.inbound_nodes[2] is None:
            # No bias term
            self.value = np.dot(X, W)
        else:
            b = self.inbound_nodes[2].value
            self.value = np.dot(X, W) + b

    def backward(self):
        """Compute the backward propagation of the node."""
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            # Find current grad
            grad_cost = n.gradients[self]

            # Compute partial grads
            dX = np.dot(grad_cost, self.inbound_nodes[1].value.T)
            dW = np.dot(self.inbound_nodes[0].value.T, grad_cost)
            cumul_b = np.sum(grad_cost, axis=0, keepdims=False)

            # Add to current node grads
            self.gradients[self.inbound_nodes[0]] += dX
            self.gradients[self.inbound_nodes[1]] += dW
            self.gradients[self.inbound_nodes[2]] += cumul_b

            # TEST
            #print("LINEAR OK")
