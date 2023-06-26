import numpy as np
import networkx as nx
from models.aggregation_dissemination import AggregationDissemination
from models.basic_majority import BasicMajority

class GeneralAggregationDissemination(BasicMajority):
    def __init__(self, theta, q, p, vg, vb, network, seed, r, k):
        super().__init__(theta, q, p, vg, vb, network)
        self.seed = seed
        self.r = r
        self.k = k

    def find_r_neighborhood(self, r, node):
        return nx.ego_graph(self.network, node, r, center=False)

    def find_k_indep(self, neighborhood, k):
        maximal_indep_set = np.array(nx.maximal_independent_set(neighborhood))
        k = k if len(maximal_indep_set) > k else len(maximal_indep_set)
        random_indices = np.random.choice(len(maximal_indep_set), k, replace=False)
        return maximal_indep_set[random_indices]

    def get_seed(self, k):
        nodes = np.array(self.network.nodes)
        random_indices = np.random.choice(len(nodes), k, replace=False)
        seed_nodes = nodes[random_indices]
        for agent in seed_nodes:
            self.network.nodes[agent]["high value"] = 1
        return seed_nodes

    def make_decisions(self, ordering, high_val):
        seed_nodes = self.get_seed(self.seed)

        for agent in seed_nodes:
            neighborhood = self.find_r_neighborhood(self.r, agent)
            k_indep_neighbors = self.find_k_indep(neighborhood, self.k)

            for indep_neighbor in k_indep_neighbors:
                if self.network.nodes[indep_neighbor]["action"] == -1:
                    self.network.nodes[indep_neighbor]["action"] = np.random.choice([self.theta, 1 - self.theta], p=[self.q, 1 - self.q])
                    self.indep_dec += 1

            neighbors = k_indep_neighbors
            actions = nx.get_node_attributes(self.network, "action")
            n_actions = np.array([actions[key] for key in neighbors])
            choice = np.random.choice([self.theta, 1 - self.theta], p=[self.q, 1 - self.q])

            zeros = n_actions[n_actions == 0]
            ones = n_actions[n_actions == 1]

            if len(neighbors) < 2 or (len(n_actions) - np.count_nonzero(n_actions == -1)) < 2:
                # payoff = vg * ((p*signal)/(p*signal + (1-p)*(1-signal))) + vb * ((1-p)*(1-signal)/(p*signal + (1-p)*(1-signal)))
                self.network.nodes[agent]["action"] = choice
                self.indep_dec += 1
            elif len(zeros) - len(ones) > 1:
                self.network.nodes[agent]["action"] = 0
            elif len(ones) - len(zeros) > 1:
                self.network.nodes[agent]["action"] = 1
            else:
                # payoff = vg * ((p * signal) / (p * signal + (1 - p) * (1 - signal))) + vb * (
                #            (1 - p) * (1 - signal) / (p * signal + (1 - p) * (1 - signal)))
                self.network.nodes[agent]["action"] = choice
                self.indep_dec += 1

        for agent in ordering:
            if self.network.nodes[agent]["action"] == -1:
                if high_val:
                    # Get the neighbors of the agent
                    neighbors = np.array(list(self.network.neighbors(agent)))

                    # Filter neighbors that are high degree and have taken action
                    valid_neighbors = np.array([n for n in neighbors if (
                                (self.network.nodes[n]["high value"] == 1) and (
                                    self.network.nodes[n]["action"] != -1))])

                    if len(valid_neighbors) > 0:
                        # Sort valid neighbors by degree in descending order
                        valid_neighbors_degrees = np.array([self.network.degree[n] for n in valid_neighbors])
                        sorted_indices = np.argsort(valid_neighbors_degrees)[::-1]
                        sorted_valid_neighbors = valid_neighbors[sorted_indices]

                        # Set agent's action to the action of the highest degree valid neighbor
                        highest_degree_neighbor = sorted_valid_neighbors[0]
                        self.network.nodes[agent]["action"] = self.network.nodes[highest_degree_neighbor]["action"]
                        self.network.nodes[agent]["high value"] = 1

                        # try majority rule for
                    else:
                        self.make_decision(agent)
                else:
                    self.make_decision(agent)
