import numpy as np
import networkx as nx
from models.basic_majority import BasicMajority

class BasicAggregation(BasicMajority):
    def __init__(self, theta, q, p, vg, vb, network, k):
        super().__init__(theta, q, p, vg, vb, network)
        self.k = k

    def find_nodes_greater_than_logn(self):
        threshold = np.log2(len(self.network.nodes()))
        nodes = np.array(list(self.network.nodes()))
        degrees = np.array(list(self.network.degree()), dtype=int)[:, 1]
        res = nodes[degrees >= threshold]

        return res.tolist()

    def order_nodes_by_degree(self, nodes):
        sorted_degrees = np.sort(nodes)[::-1]
        return sorted_degrees
    def find_k_rand_neighbors(self, node, k):
        neighbors = np.array(list(self.network.neighbors(node)))
        random_indices = np.random.choice(len(neighbors), k, replace=False)

        return neighbors[random_indices]

    def make_decisions(self, ordering):

        #find all nodes with deg >= logn
        high_val_nodes = self.find_nodes_greater_than_logn()

        #order by degree (descending)
        high_val_nodes = self.order_nodes_by_degree(high_val_nodes)

        #for all nodes in set
        for agent in high_val_nodes:

            #find k random neighbors
            k_rand_neighbors = self.find_k_rand_neighbors(agent, self.k)

            #have k random neighbors make decisions
            for neighbor in k_rand_neighbors:
                if self.network.nodes[neighbor]["action"] == -1:
                    self.make_decision(neighbor)

            #aggregate
            self.make_decision(agent)

        #sim rest of the nodes (random ordering)
        for agent in ordering:
            if self.network.nodes[agent]["action"] == -1:
                self.make_decision(agent)
