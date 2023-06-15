import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from models.basic_majority import BasicMajority
from models.aggregation_dissemination import AggregationDissemination

#parameters
n = 500
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
watts_strogatz = nx.connected_watts_strogatz_graph(n, 6, 0.1, tries=100, seed=None)
prob_edge_creation = 0.01
erdos = nx.erdos_renyi_graph(n, prob_edge_creation, seed=None, directed=False)
power_law = nx.barabasi_albert_graph(n, 3, seed=None, initial_graph=None)

# nx.draw_spring(three_ary_tree, with_labels=True)
# plt.show()


def run_sim(n, p, q, vg, vb, theta, network):
    index = np.around(np.linspace(0, n-1, n)).astype(int)
    priv_signal = np.full(n, q)
    priv_signals = dict(zip(index, priv_signal))
    action = np.full(n, -1)
    actions = dict(zip(index, action))
    nx.set_node_attributes(network, priv_signals, "private signal")
    nx.set_node_attributes(network, actions, "action")

    #create a random node ordering
    ordering = np.arange(n-1,-1,-1)

    basic_majority_model = BasicMajority(theta, q, p, vg, vb, network)
    basic_majority_model.make_decisions(ordering)

    #final measurements
    return basic_majority_model.calc_success_rate()

# set node information
def run_sim_aggregate(n, p, q, vg, vb, theta, network, hi, lo):
    index = np.around(np.linspace(0, n-1, n)).astype(int)
    priv_signal = np.full(n, q)
    priv_signals = dict(zip(index, priv_signal))
    action = np.full(n, -1)
    actions = dict(zip(index, action))
    nx.set_node_attributes(network, priv_signals, "private signal")
    nx.set_node_attributes(network, actions, "action")

    #create a random node ordering
    ordering = np.arange(n-1,-1,-1)

    #make decisions
    agg_dis_model = AggregationDissemination(theta, q, p, vg, vb, network, hi, lo)
    agg_dis_model.make_decisions(ordering)

    return agg_dis_model.calc_success_rate()

#results
num_trials = 100
total_runs = np.zeros(num_trials)

for i in range(0, num_trials):
    total_runs[i] = run_sim(n, p, q, vg, vb, theta, power_law)

total_runs = np.array(total_runs)
print("\nRegular")
print(f"average learning rate: {np.mean(total_runs)}")
print(f"Learning rate range: [{np.min(total_runs)}, {np.max(total_runs)}]")
print(f"less than q: {len(total_runs[total_runs < q])}")

for i in range(0, num_trials):
    total_runs[i] = run_sim_aggregate(n, p, q, vg, vb, theta, power_law, 10, 10)

total_runs = np.array(total_runs)
print("\nAggregate")
print(f"average learning rate: {np.mean(total_runs)}")
print(f"Learning rate range: [{np.min(total_runs)}, {np.max(total_runs)}]")
print(f"less than q: {len(total_runs[total_runs < q])}")



# Compute the histogram
#hist, bins = np.histogram(total_runs, bins=10)

# Plot the histogram
#plt.hist(total_runs, bins=10)
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Histogram')
#plt.show()





