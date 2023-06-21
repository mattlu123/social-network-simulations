import numpy as np
import networkx as nx
from models.basic_majority import BasicMajority

class AggregationDissemination(BasicMajority):
    def __init__(self, theta, q, p, vg, vb, network, hi_percentile, lo_percentile):
        super().__init__(theta, q, p, vg, vb, network)
        self.hi_percentile = hi_percentile
        self.lo_percentile = lo_percentile

    def get_top_percentile_nodes(self, high_val):
        # Get the degrees of all nodes in the graph
        degrees = dict(self.network.degree())

        # Calculate the threshold degree for the given percentile
        sorted_degrees = sorted(degrees.values(), reverse=True)
        threshold_degree = sorted_degrees[int(len(sorted_degrees) * self.hi_percentile / 100)]

        # Find the nodes with degrees above the threshold
        top_percentile_nodes = [node for node, degree in degrees.items() if degree >= threshold_degree]

        if high_val:
            for agent in top_percentile_nodes:
                self.network.nodes[agent]['high_value'] = 1

        return top_percentile_nodes

    def get_lowest_percentile_neighbors(self, node):
        # Get the degrees of all neighbors of the node
        degrees = dict(self.network.degree(self.network.neighbors(node)))

        # Calculate the threshold degree for the given percentile
        sorted_degrees = sorted(degrees.values())
        threshold_degree = sorted_degrees[int(len(sorted_degrees) * self.lo_percentile / 100)]

        # Find the neighbors with degrees below the threshold
        lowest_percentile_neighbors = [n for n, degree in degrees.items() if degree <= threshold_degree]

        return np.random.permutation(lowest_percentile_neighbors)

    def make_decisions(self, ordering, high_val):
        highDegreeNodes = self.get_top_percentile_nodes(high_val)
        for agent in highDegreeNodes:
            lowDegreeNeighbors = self.get_lowest_percentile_neighbors(agent)

            for Low_agent in lowDegreeNeighbors:
                if self.network.nodes[Low_agent]["action"] == -1:
                    self.make_decision(Low_agent)

            neighbors = lowDegreeNeighbors
            actions = nx.get_node_attributes(self.network, "action")
            n_actions = [actions[key] for key in neighbors]
            choice = np.random.choice([self.theta, 1 - self.theta], p=[self.q, 1 - self.q])

            zeros = [num for num in n_actions if num == 0]
            ones = [num for num in n_actions if num == 1]

            if len(neighbors) < 2 or (len(n_actions) - n_actions.count(-1)) < 2:
                # payoff = vg * ((p*signal)/(p*signal + (1-p)*(1-signal))) + vb * ((1-p)*(1-signal)/(p*signal + (1-p)*(1-signal)))
                self.network.nodes[agent]["action"] = choice
            elif len(zeros) - len(ones) > 1:
                self.network.nodes[agent]["action"] = 0
            elif len(ones) - len(zeros) > 1:
                self.network.nodes[agent]["action"] = 1
            else:
                # payoff = vg * ((p * signal) / (p * signal + (1 - p) * (1 - signal))) + vb * (
                #            (1 - p) * (1 - signal) / (p * signal + (1 - p) * (1 - signal)))
                self.network.nodes[agent]["action"] = choice

                # run simulation
        for agent in ordering:
            if self.network.nodes[agent]["action"] == -1:
                if high_val:
                    # look for high degree neighbor that has already taken action
                    degree_sorted_neighbors = sorted(dict(self.network.degree(self.network.neighbors(agent))), reverse=True)
                    for neighbor in degree_sorted_neighbors:
                        if (self.network.nodes[neighbor]["high value"] == 1 and self.network.nodes[neighbor]["action"] != -1):
                            self.network.nodes[agent]["action"] = self.network.nodes[neighbor]["action"]
                            break
                self.make_decision(agent)
