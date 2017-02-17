from Node import Node
import numpy as np

class CrossEntropy(Node):
    def __init__(self, y, y_hat, reg=0.):
        """A node that represents the softmax loss.
        Should always be last node in a computational graph.
        Only useful for multinomial classification problems.
        """
        Node.__init__(self, [y_hat])

        self.y = y
        self.reg = reg

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def forward(self):
        """Forward propagate the node value."""
        y = self.y.value
        y_hat = self._sigmoid(self.inbound_nodes[0].value)

        m = y.shape[0] # Num training examples

        # Compute unregularized loss
        f = y * np.log(y_hat)
        s = (1 - y) * np.where(1 - y_hat > 1e-6, np.log(1 - y_hat), 0.) # Prevents div by 0.

        # Compute regularization term
        W = self.inbound_nodes[0].inbound_nodes[1].value # Weights
        r = self.reg / (2*m) * np.sum(W * W)

        loss = -1./m * np.sum(np.sum(f + s)) + r
        self.value = loss

    def backward(self):
        """Backpropagate gradients."""
        delta = self._sigmoid(self.inbound_nodes[0].value) - self.y.value

        self.gradients[self.inbound_nodes[0]] = delta
