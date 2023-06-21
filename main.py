import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from models.basic_majority import BasicMajority
from models.aggregation_dissemination import AggregationDissemination
from utils import sim_n_times, plot_histograms

#parameters
n = 1000 #number of agents
p = 0.5 #ground truth probability
q = 0.7 #private signal confidence
vg, vb = 1, -1 #payoff coefficients
theta = np.random.choice([0, 1], p=[p, 1-p]) #ground truth

#graph construction
tree = nx.random_tree(n=n, seed=0)
b_factor = 3
three_ary_tree = nx.full_rary_tree(b_factor, n)
m, k = 5, 100
bipartite = nx.complete_multipartite_graph(m, k)
watts_strogatz = nx.connected_watts_strogatz_graph(n, 6, 0.1, tries=100, seed=None)
prob_edge_creation = 0.01
erdos = nx.erdos_renyi_graph(n, prob_edge_creation, seed=None, directed=False)
k = 20
clique = nx.complete_graph(k)
power_law = nx.barabasi_albert_graph(n, k, seed=None, initial_graph=clique)

# nx.draw_spring(three_ary_tree, with_labels=True)
# plt.show()

def init_net(n, q, network):
    index = np.around(np.linspace(0, n - 1, n)).astype(int)

    #fixed private signal for all agents
    priv_signal = np.full(n, q)
    priv_signals = dict(zip(index, priv_signal))
    nx.set_node_attributes(network, priv_signals, "private signal")

    #set initial actions to -1
    action = np.full(n, -1)
    actions = dict(zip(index, action))
    nx.set_node_attributes(network, actions, "action")

    #set initial hi or lo values
    high_value = dict(zip(index, np.full(n, -1)))
    nx.set_node_attributes(network, high_value, "high value")

def run_sim(n, p, q, vg, vb, theta, network):
    #initialize network params
    init_net(n, q, network)

    #create a random node ordering
    ordering = np.random.permutation(np.arange(0, n))

    #construct the model
    basic_majority_model = BasicMajority(theta, q, p, vg, vb, network)
    basic_majority_model.make_decisions(ordering)

    #final measurements
    return basic_majority_model.calc_success_rate(), basic_majority_model.get_indep_decisions()

# set node information
def run_sim_aggregate(n, p, q, vg, vb, theta, network, hi, lo):

    #initialize network params
    init_net(n, q, network)

    #create a random node ordering
    ordering = np.random.permutation(np.arange(0, n))

    #make decisions
    agg_dis_model = AggregationDissemination(theta, q, p, vg, vb, network, hi, lo)
    agg_dis_model.make_decisions(ordering, False)

    return agg_dis_model.calc_success_rate(), agg_dis_model.get_indep_decisions()

#results
basic_majority_res = sim_n_times("Basic Majority Model", 100, run_sim,
                                 (n, p, q, vg, vb, theta, power_law))
aggregation_res = sim_n_times("Aggregation Model", 100, run_sim_aggregate,
                              (n, p, q, vg, vb, theta, power_law, 1, 1))
plot_histograms("Basic Majority Model", basic_majority_res)
plot_histograms("Aggregation Model", aggregation_res)








# deltas = [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
# k = 0
# sum = 0
#
# res = np.zeros(len(deltas))
# for idx, delta in enumerate(deltas):
#     k = 0
#     sum = 0
#     while sum < 1 - delta:
#         sum = 1 - stats.binom.cdf(k // 2, k, q)
#         k += 1
#     res[idx] = k
#     print(f"delta = {delta}: {k}")

