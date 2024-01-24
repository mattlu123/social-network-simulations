import numpy as np
import networkx as nx
from models.basic_majority import BasicMajority
import copy

class TwoNeighbors(BasicMajority):

    def __init__(self, theta, q, p, vg, vb, network, high_value):
        super().__init__(theta, q, p, vg, vb, network)
        self.high_value = high_value

    def make_decisions(self, ordering):

        all_nodes = set(ordering)

        if self.high_value:

            # Pick two high degree nodes to be high value
            agent_1 = all_nodes.pop()
            agent_2 = all_nodes.pop()
            neighbors_1 = self.network.neighbors(agent_1)
            neighbors_2 = self.network.neighbors(agent_2)

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

        while all_nodes:

            agent = all_nodes.pop()
            neighbors = self.network.neighbors(agent)

            for neighbor in neighbors:
                self.make_decision(neighbor)

                if neighbor in all_nodes:
                    all_nodes.remove(neighbor)

            self.make_decision(agent)
            if self.high_value:
                self.network.nodes[agent]["high value"] = 1

            temp = copy.deepcopy(all_nodes)
            while temp:
                node = temp.pop()
                count = 0
                actions = nx.get_node_attributes(self.network, "action")
                quality = nx.get_node_attributes(self.network, "high value")

                for n in (self.network.neighbors(node)):
                    if self.high_value:
                        count += 1 if (actions[n] != -1 and quality[n] != -1) else 0
                    else:
                        count += 1 if actions[n] != -1 else 0
                if count >= 2:
                    self.make_decision(node)
                    self.network.nodes[node]["high value"] = 1
                    all_nodes.remove(node)

