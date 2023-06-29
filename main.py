import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from models.basic_majority import BasicMajority
from models.aggregation_dissemination import AggregationDissemination
from models.general_agg_diss import GeneralAggregationDissemination
from utils import sim_n_times, plot_histograms, plot, plot_heatmap

#parameters
n = 1000 #number of agents
p = 0.5 #ground truth probability
q = 0.7 #private signal confidence
vg, vb = 1, -1 #payoff coefficients
theta = np.random.choice([0, 1], p=[p, 1-p]) #ground truth

#graph construction
tree = nx.random_tree(n=n, seed=0)
b_factor = 4
r_ary_tree = nx.full_rary_tree(b_factor, n)
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
    agg_dis_model.make_decisions(ordering, True)

    return agg_dis_model.calc_success_rate(), agg_dis_model.get_indep_decisions()

def run_sim_general_aggregate(n, p, q, vg, vb, theta, network, seed, r, k):

    #initialize network params
    init_net(n, q, network)

    #create a random node ordering
    ordering = np.random.permutation(np.arange(0, n))
    
    

    #make decisions
    gen_agg_dis_model = GeneralAggregationDissemination(theta, q, p, vg, vb, network, seed, r, k)
    gen_agg_dis_model.make_decisions(ordering, True)

    return gen_agg_dis_model.calc_success_rate(), gen_agg_dis_model.get_indep_decisions()

#results
# basic_majority_res = sim_n_times("Basic Majority Model", 1000, run_sim,
#                                 (n, p, q, vg, vb, theta, erdos))
# # aggregation_res = sim_n_times("Aggregation Model", 1000, run_sim_aggregate,
# #                             (n, p, q, vg, vb, theta, power_law, 0.001, 2))
# r = 5
# k = 40
# seed = 2
# gen_agg_res = sim_n_times("General Aggregation Model", 1000, run_sim_general_aggregate,
#                           (n, p, q, vg, vb, theta, erdos, seed, r, k))

rs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
seeds = [2, 4, 6, 8, 10, 12]

res = np.zeros((len(seeds), len(rs)))
for i, seed in enumerate(seeds):
    res_rad = np.zeros(len(rs))
    for j, rad in enumerate(rs):
        avg_learn_rate = np.mean(sim_n_times("General Aggregation Model", 100, run_sim_general_aggregate,
                                             (n, p, q, vg, vb, theta, erdos, seed, rad, k)))
        res_rad[j] = avg_learn_rate
    res[i] = res_rad

plot_heatmap(res, "radius", "seed", rs, seeds, "Avg learning rate with diff radius and seed vals")