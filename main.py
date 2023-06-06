import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

#parameters
n = 1000
p = 0.5
q = 0.9
vg, vb = 1, -1
theta = np.random.choice([0, 1], p=[p, 1-p])

#graph construction
tree = nx.random_tree(n=n, seed=0)

#set node information
index = np.around(np.linspace(0, n-1, n)).astype(int)
priv_signal = np.full(n, q)
priv_signals = dict(zip(index, priv_signal))
action = np.full(n, -1)
actions = dict(zip(index, action))
nx.set_node_attributes(tree, priv_signals, "private signal")
nx.set_node_attributes(tree, actions, "action")

#create a random node ordering
ordering = np.random.permutation(np.arange(0, n))

#run simulation
for agent in ordering:
    signal = nx.get_node_attributes(tree, "private signal")[agent]
    neighbors = [n for n in tree.neighbors(agent)]
    actions = nx.get_node_attributes(tree, "action")
    n_actions = [actions[key] for key in neighbors]
    choice = np.random.choice([theta, 1 - theta], p=[q, 1 - q])

    if len(neighbors) < 2 or (len(n_actions) - n_actions.count(-1)) < 2:
        payoff = vg * ((p*signal)/(p*signal + (1-p)*(1-signal))) + vb * ((1-p)*(1-signal)/(p*signal + (1-p)*(1-signal)))
        tree.nodes[agent]["action"] = choice if payoff > 0 else 1-theta
        continue

    zeros = [num for num in n_actions if num == 0]
    ones = [num for num in n_actions if num == 0]
    if len(zeros) - len(ones) > 1:
        tree.nodes[agent]["action"] = 0
    elif len(ones) - len(zeros) > 1:
        tree.nodes[agent]["action"] = 1
    else:
        payoff = vg * ((p * signal) / (p * signal + (1 - p) * (1 - signal))) + vb * (
                    (1 - p) * (1 - signal) / (p * signal + (1 - p) * (1 - signal)))
        tree.nodes[agent]["action"] = choice if payoff > 0 else 1 - theta

#final measurements
final_actions = np.array(list(nx.get_node_attributes(tree, "action").values()))
success_rate = len(final_actions[final_actions == theta])/len(final_actions)
print(success_rate)

# nx.draw_spring(tree, with_labels=True)
# plt.show()




