import numpy as np
import networkx as nx
from models.basic_majority import BasicMajority
import copy

class TwoNeighborsHighVal(BasicMajority):

    def __init__(self, theta, q, p, vg, vb, network, threshold, k):
        super().__init__(theta, q, p, vg, vb, network)
        self.threshold = threshold
        self.k = k

    def make_decisions(self, ordering):

        all_nodes = set(ordering)
        index = 2

        # Pick two random nodes to be high value
        agent_1 = ordering[0]
        agent_2 = ordering[1]
        all_nodes.remove(agent_1)
        all_nodes.remove(agent_2)
        neighbors_1 = self.find_r_neighborhood(1, agent_1)
        neighbors_2 = self.find_r_neighborhood(1, agent_2)
        neighbors_1 = self.find_k_indep(neighbors_1, self.k)
        neighbors_2 = self.find_k_indep(neighbors_2, self.k)

        for neighbor in neighbors_1:
            self.make_decision(neighbor)

            if neighbor in all_nodes:
                all_nodes.remove(neighbor)

        self.make_decision(agent_1)
        self.network.nodes[agent_1]["high value"] = 1

        for neighbor in neighbors_2:
            self.make_decision(neighbor)

            if neighbor in all_nodes:
                all_nodes.remove(neighbor)

        # process rest of the nodes
        while all_nodes:
            while index < len(ordering) and self.network.nodes[ordering[index]]["action"] != -1:
                index += 1
            agent = ordering[index]
            all_nodes.remove(agent)
            neighbors = self.find_r_neighborhood(1, agent)
            neighbors = self.find_k_indep(neighbors, self.k) if neighbors.number_of_nodes() > 0 else []

            for neighbor in neighbors:
                self.make_decision(neighbor)
                self.network.nodes[neighbor]["high value"] = 0

                if neighbor in all_nodes:
                    all_nodes.remove(neighbor)

            self.make_decision(agent)
            self.network.nodes[agent]["high value"] = 1

            temp = copy.deepcopy(all_nodes)

            while temp:
                node = temp.pop()
                hi_quality_neighbors = 0
                lo_quality_neighbors = 0
                count = 0
                actions = nx.get_node_attributes(self.network, "action")
                values = nx.get_node_attributes(self.network, "high value")
                for n in (self.network.neighbors(node)):
                    hi_quality_neighbors += 1 if values[n] == 1 else 0
                    lo_quality_neighbors += 1 if values[n] == 0 else 0
                    count += 1 if actions[n] != -1 else 0
                #print("degree: ", self.network.degree(node))
                #print("hi: ", hi_quality_neighbors)
                #print("lo: ", lo_quality_neighbors)
                #print()
                quality = -1
                if hi_quality_neighbors + lo_quality_neighbors != 0:
                    quality = hi_quality_neighbors / (hi_quality_neighbors + lo_quality_neighbors)
                if count >= 2 and quality > self.threshold:
                    #print("quality: ", quality)
                    self.make_decision(node)
                    self.network.nodes[node]["high value"] = 1
                    all_nodes.remove(node)
