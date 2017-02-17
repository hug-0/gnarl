class Node(object):
    def __init__(self, inbound_nodes=[]):
        """A base node in a computational graph."""
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.value = None # Init first value as None
        self.gradients = {}

        # Append this node to all nodes that point to it
        for in_node in self.inbound_nodes:
            in_node.outbound_nodes.append(self)

    def forward(self):
        """Forward propagate input from inbound nodes to outbound nodes."""
        raise NotImplemented

    def backward(self):
        """Backward propagate output from outbound nodes to inbound nodes."""
        raise NotImplemented
