import numpy as np
import networkx as nx

class BasicMajority:
    def __init__(self, theta, q, p, vg, vb, network):
        self.theta = theta
        self.q = q
        self.p = p
        self.vg = vg
        self.vb = vb
        self.network = network
        self.indep_dec = 0

    #Make a single decision
    def make_decision(self, agent):
        if self.network.nodes[agent]["action"] != -1:
            return
        
        #signal = nx.get_node_attributes(self.network, "private signal")[agent]
        neighbors = np.array([n for n in self.network.neighbors(agent)])
        actions = nx.get_node_attributes(self.network, "action")
        n_actions = np.array([actions[key] for key in neighbors])
        choice = np.random.choice([self.theta, 1 - self.theta], p=[self.q, 1 - self.q])

        if len(neighbors) < 2 or (len(n_actions) - np.count_nonzero(n_actions == -1)) < 2:
            self.network.nodes[agent]["action"] = choice
            self.indep_dec += 1
            return

        zeros = n_actions[n_actions == 0]
        ones = n_actions[n_actions == 1]
        if len(zeros) - len(ones) > 1:
            self.network.nodes[agent]["action"] = 0
        elif len(ones) - len(zeros) > 1:
            self.network.nodes[agent]["action"] = 1
        else:
            self.network.nodes[agent]["action"] = choice

    def make_decisions(self, ordering):
        for agent in ordering:
            if self.network.nodes[agent]["action"] == -1:
                self.make_decision(agent)

    def calc_payoff(self, signal):
        payoff = self.vg * ((self.p * signal) / (self.p * signal + (1 - self.p) * (1 - signal))) + self.vb * (
                (1 - self.p) * (1 - signal) / (self.p * signal + (1 - self.p) * (1 - signal)))
        return payoff

    def calc_success_rate(self):
        final_actions = np.array(list(nx.get_node_attributes(self.network, "action").values()))
        success_rate = len(final_actions[final_actions == self.theta]) / len(final_actions)
        return success_rate

    def get_indep_decisions(self):
        return self.indep_dec