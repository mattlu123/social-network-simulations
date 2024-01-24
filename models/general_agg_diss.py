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

    def get_seed(self, k):
        # nodes = np.array(self.network.nodes)
        # random_indices = np.random.choice(len(nodes), k, replace=False)
        # seed_nodes = nodes[random_indices]
        seed_nodes = np.array(nx.maximal_independent_set(self.network))
        random_indices = np.random.choice(len(seed_nodes), k, replace=False)
        seed_nodes = seed_nodes[random_indices]
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
                    #self.make_decision(indep_neighbor)

            neighbors = k_indep_neighbors
            actions = nx.get_node_attributes(self.network, "action")
            n_actions = np.array([actions[key] for key in neighbors])
            choice = np.random.choice([self.theta, 1 - self.theta], p=[self.q, 1 - self.q])

            zeros = n_actions[n_actions == 0]
            ones = n_actions[n_actions == 1]

            if len(neighbors) < 2 or (len(n_actions) - np.count_nonzero(n_actions == -1)) < 2:
                self.network.nodes[agent]["action"] = choice
                self.indep_dec += 1
            elif len(zeros) - len(ones) > 1:
                self.network.nodes[agent]["action"] = 0
            elif len(ones) - len(zeros) > 1:
                self.network.nodes[agent]["action"] = 1
            else:
                self.network.nodes[agent]["action"] = choice
                self.indep_dec += 1
                
        #return back to neighbors of high degree nodes to go first
        #BFS from high value nodes
        # for agent in seed_nodes:
        #     visited = set()  # To keep track of visited nodes
        #     queue = [agent]  # Queue for BFS traversal
        #
        #     while queue:
        #         node = queue.pop(0)
        #         visited.add(node)
        #
        #         neighbors = self.network.neighbors(node)
        #         for neighbor in neighbors:
        #             if neighbor not in visited:
        #                 queue.append(neighbor)
        #                 visited.add(neighbor)
        #                 AggregationDissemination.make_high_value_decision(self, neighbor)

        for agent in ordering:
            if high_val:
                AggregationDissemination.make_high_value_decision(self, agent)
            else:
                self.make_decision(agent)
