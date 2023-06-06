import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

#parameters
n = 28
p = 0.5
q = 0.8
vg, vb = 1, -1
theta = np.random.choice([0, 1], p=[p, 1-p])

#graph construction
tree = nx.random_tree(n=n, seed=0)
b_factor = 3
three_ary_tree = nx.full_rary_tree(b_factor, n)
m, k = 5, 100
bipartite = nx.complete_multipartite_graph(m, k)

# nx.draw_spring(bipartite, with_labels=False)
# plt.show()

#set node information
def run_sim(n, p, q, vg, vb, theta, network):
    index = np.around(np.linspace(0, n-1, n)).astype(int)
    priv_signal = np.full(n, q)
    priv_signals = dict(zip(index, priv_signal))
    action = np.full(n, -1)
    actions = dict(zip(index, action))
    nx.set_node_attributes(network, priv_signals, "private signal")
    nx.set_node_attributes(network, actions, "action")

    #create a random node ordering
    ordering = np.random.permutation(np.arange(0, n))

    #run simulation
    for agent in ordering:
        signal = nx.get_node_attributes(network, "private signal")[agent]
        neighbors = [n for n in network.neighbors(agent)]
        actions = nx.get_node_attributes(network, "action")
        n_actions = [actions[key] for key in neighbors]
        choice = np.random.choice([theta, 1 - theta], p=[q, 1 - q])

        if len(neighbors) < 2 or (len(n_actions) - n_actions.count(-1)) < 2:
            payoff = vg * ((p*signal)/(p*signal + (1-p)*(1-signal))) + vb * ((1-p)*(1-signal)/(p*signal + (1-p)*(1-signal)))
            network.nodes[agent]["action"] = choice if payoff > 0 else 1-theta
            continue

        zeros = [num for num in n_actions if num == 0]
        ones = [num for num in n_actions if num == 0]
        if len(zeros) - len(ones) > 1:
            network.nodes[agent]["action"] = 0
        elif len(ones) - len(zeros) > 1:
            network.nodes[agent]["action"] = 1
        else:
            payoff = vg * ((p * signal) / (p * signal + (1 - p) * (1 - signal))) + vb * (
                        (1 - p) * (1 - signal) / (p * signal + (1 - p) * (1 - signal)))
            network.nodes[agent]["action"] = choice if payoff > 0 else 1 - theta

    #final measurements
    final_actions = np.array(list(nx.get_node_attributes(network, "action").values()))
    success_rate = len(final_actions[final_actions == theta])/len(final_actions)
    return success_rate

#results
num_trials = 1000
total_runs = np.zeros(num_trials)
for i in range(0, num_trials):
    total_runs[i] = run_sim(m+k, p, q, vg, vb, theta, bipartite)

print(f"average learning rate: {np.mean(total_runs)}")
print(f"Learning rate range: [{np.min(total_runs)}, {np.max(total_runs)}]")




