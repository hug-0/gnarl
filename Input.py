from Node import Node
import numpy as np

class Input(Node):
    def __init__(self):
        """An input node. Input nodes don't perform any computations.
        Rather, they represent the input features that will be fed into
        the neural network.
        """
        Node.__init__(self)

    def forward(self, value=None):
        """Forward propagate input value"""
        if value is not None:
            self.value = value

    def backward(self):
        """Backward propagate from outbound nodes to this node."""
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost

            # TEST
            #print("INPUT OK")
