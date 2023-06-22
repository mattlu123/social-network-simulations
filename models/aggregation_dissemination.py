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
        nodes = np.array(list(self.network.nodes()))
        degrees = np.array(list(self.network.degree()), dtype=int)[:, 1]  # Extract the degree values

        # Calculate the threshold degree for the given percentile
        sorted_degrees = np.sort(degrees)[::-1]
        threshold_index = int(len(sorted_degrees) * self.hi_percentile / 100)
        threshold_degree = sorted_degrees[threshold_index]

        # Find the nodes with degrees above the threshold
        top_percentile_nodes = nodes[degrees >= threshold_degree]

        if high_val:
            for agent in top_percentile_nodes:
                self.network.nodes[agent]['high_value'] = 1

        return top_percentile_nodes

    def get_lowest_percentile_neighbors(self, node):
        # Get the neighbors of the node
        neighbors = np.array(list(self.network.neighbors(node)))

        # Get the degrees of all neighbors of the node
        degrees = np.array([self.network.degree[n] for n in neighbors])

        # Calculate the threshold degree for the given percentile
        sorted_degrees = np.sort(degrees)
        threshold_index = int(len(sorted_degrees) * self.lo_percentile / 100)
        threshold_degree = sorted_degrees[threshold_index]

        # Find the neighbors with degrees below the threshold
        lowest_percentile_neighbors = neighbors[degrees <= threshold_degree]

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
            n_actions = np.array([actions[key] for key in neighbors])
            choice = np.random.choice([self.theta, 1 - self.theta], p=[self.q, 1 - self.q])

            zeros = n_actions[n_actions == 0]
            ones = n_actions[n_actions == 1]

            if len(neighbors) < 2 or (len(n_actions) - np.count_nonzero(n_actions == -1)) < 2:
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
                    # Get the neighbors of the agent
                    neighbors = np.array(list(self.network.neighbors(agent)))

                    # Filter neighbors that are high degree and have taken action
                    valid_neighbors = neighbors[
                        (self.network.nodes[neighbors]["high value"] == 1) &
                        (self.network.nodes[neighbors]["action"] != -1)
                    ]

                    if len(valid_neighbors) > 0:
                        # Sort valid neighbors by degree in descending order
                        valid_neighbors_degrees = np.array([self.network.degree[n] for n in valid_neighbors])
                        sorted_indices = np.argsort(valid_neighbors_degrees)[::-1]
                        sorted_valid_neighbors = valid_neighbors[sorted_indices]

                        # Set agent's action to the action of the highest degree valid neighbor
                        highest_degree_neighbor = sorted_valid_neighbors[0]
                        self.network.nodes[agent]["action"] = self.network.nodes[highest_degree_neighbor]["action"]

                self.make_decision(agent)
