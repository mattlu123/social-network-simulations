import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from models.basic_majority import BasicMajority
from models.aggregation_dissemination import AggregationDissemination
from models.general_agg_diss import GeneralAggregationDissemination
from models.basic_aggregation import BasicAggregation
from models.two_neighbors import TwoNeighbors
from models.two_neighbors_high_val import TwoNeighborsHighVal
from utils import sim_n_times, plot_histograms, plot, plot_heatmap
import time

#parameters
n = 300 #number of agents
p = 0.5 #ground truth probability
q = 0.7 #private signal confidence
vg, vb = 1, -1 #payoff coefficients
theta = np.random.choice([0, 1], p=[p, 1-p]) #ground truth

#graph construction
tree = nx.random_tree(n=511, seed=0)
b_factor = 4
r_ary_tree = nx.full_rary_tree(b_factor, n)
m, k = 5, 100
bipartite = nx.complete_multipartite_graph(m, k)
watts_strogatz = nx.connected_watts_strogatz_graph(n, 6, 0.1, tries=100, seed=None)
prob_edge_creation = 0.01
erdos = nx.erdos_renyi_graph(n, prob_edge_creation, seed=None, directed=False)
k = 5
clique = nx.complete_graph(k)
power_law = nx.barabasi_albert_graph(n, k, seed=None, initial_graph=clique)

# nx.draw_spring(three_ary_tree, with_labels=True)
# plt.show()

def init_net(n, q, network):
    index = list(network.nodes())

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

def random_ordering(nodes):
    ordering = nodes
    np.random.shuffle(ordering)
    return ordering

def spiral_ordering(nodes, n):
    res = []
    row_start, row_end, col_start, col_end = 0, n - 1, 0, n - 1

    while row_start <= row_end and col_start <= col_end:
        for i in range(col_start, col_end + 1):
            res.append(nodes[row_start * n + i])
        row_start += 1

        for i in range(row_start, row_end + 1):
            res.append(nodes[i * n + col_end])
        col_end -= 1

        if row_start <= row_end:
            for i in range(col_end, col_start - 1, -1):
                res.append(nodes[row_end * n + i])
        row_end -= 1

        if col_start <= col_end:
            for i in range(row_end, row_start - 1, -1):
                res.append(nodes[i * n + col_start])
        col_start += 1


    return res

def degree_ordering_increasing(network):
    nodes_by_degree = sorted(network.nodes(), key=lambda x: network.degree(x))
    return nodes_by_degree

def degree_ordering_decreasing(network):
    nodes_by_degree = sorted(network.nodes(), key=lambda x: network.degree(x), reverse=True)
    return nodes_by_degree

def run_sim(n, p, q, vg, vb, theta, network):
    #initialize network params
    init_net(n, q, network)

    #create a random node ordering
    ordering = random_ordering(list(network.nodes()))
    #ordering = list(network.nodes())

    #construct the model
    basic_majority_model = BasicMajority(theta, q, p, vg, vb, network)
    basic_majority_model.make_decisions(ordering)

    #final measurements
    return basic_majority_model.calc_success_rate(), basic_majority_model.get_indep_decisions()

def run_sim_2(n, p, q, vg, vb, theta, network, nk):
    #initialize network params
    init_net(n, q, network)

    #create a random node ordering
    ordering = spiral_ordering(list(network.nodes()), nk)

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
#basic_majority_res = sim_n_times("Basic Majority Model", 100, run_sim,
#                                 (n, p, q, vg, vb, theta, power_law))
#aggregation_res = sim_n_times("Aggregation Model", 1000, run_sim_aggregate,
#                             (n, p, q, vg, vb, theta, power_law, 0.001, 2))
# r = 5
# k = 40
# seed = 2
# gen_agg_res = sim_n_times("General Aggregation Model", 1000, run_sim_general_aggregate,
#                           (n, p, q, vg, vb, theta, erdos, seed, r, k))

# #Learning rate for diff radius and seed vals on random graphs
# rs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# seeds = [2, 4, 6, 8, 10, 12]
#
# res = np.zeros((len(seeds), len(rs)))
# for i, seed in enumerate(seeds):
#     res_rad = np.zeros(len(rs))
#     for j, rad in enumerate(rs):
#         avg_learn_rate = np.mean(sim_n_times("General Aggregation Model", 100, run_sim_general_aggregate,
#                                              (n, p, q, vg, vb, theta, erdos, seed, rad, k)))
#         res_rad[j] = avg_learn_rate
#     res[i] = res_rad
#
# plot_heatmap(res, "radius", "seed", rs, seeds, "Avg learning rate with diff radius and seed vals")

# #watts-strogatz clustering vs learning rate
# ps = np.arange(0, 1.1, 0.1)
# res = np.zeros((len(ps), 2))
#
# for i, p in enumerate(ps):
#     small_worlds = nx.connected_watts_strogatz_graph(1000, 6, p, tries=100, seed=None)
#     clustering_coeff = nx.average_clustering(small_worlds)
#     avg = np.mean(sim_n_times("Basic Majority Model", 100, run_sim, (1000, p, q, vg, vb, theta, small_worlds)))
#     res[i] = [clustering_coeff, avg]
#
# plt.figure(figsize=(8, 8)).subplots_adjust(hspace=0.6)
#
# plt.subplot(1, 2, 1)
# plt.plot(ps, res[:, 0])
# plt.xlabel("p")
# plt.ylabel("avg clustering")
# plt.title("randomness vs avg clustering")
#
# plt.subplot(1, 2, 2)
# plt.plot(ps, res[:, 1])
# plt.xlabel("p")
# plt.ylabel("avg learning rate")
# plt.title("randomness vs learning rate")
#
# plt.show()

# Spiral matrix learning
# ns = np.arange(5, 51, 5)
# res_spiral = np.zeros(len(ns))
# res_reg = np.zeros(len(ns))
#
# for i, nk in enumerate(ns):
#     grid = nx.grid_2d_graph(nk, nk)
#     res_spiral[i] = np.mean(sim_n_times("Basic Majority Model", 500, run_sim_2, (nk**2, p, q, vg, vb, theta, grid, nk)))
#     res_reg[i] = np.mean(sim_n_times("Basic Majority Model", 500, run_sim, (900, p, q, vg, vb, theta, grid)))
#
# plt.title("Dimension of nxn grid vs learning rate")
# plt.xlabel("dimension")
# plt.ylabel("learning rate")
# plt.plot(ns, res_spiral, color="blue", label="Spiral ordering")
# plt.plot(ns, res_reg, color="orange", label="Random ordering")
# plt.legend()
# plt.show()

# nk = 31
# grid = nx.grid_2d_graph(nk, nk)
# sim_n_times("Basic Majority Model", 500, run_sim_2, (nk**2, p, q, vg, vb, theta, grid, nk))

def spiral_ordering_expectation(nodes, n, q):
    res = np.zeros((n, n))
    row_start, row_end, col_start, col_end = 0, n - 1, 0, n - 1
    ev = 0

    while row_start <= row_end and col_start <= col_end:
        for i in range(col_start, col_end + 1):
            curr_x = nodes[row_start * n + i][0]
            curr_y = nodes[row_start * n + i][1]
            if row_start == 0:
                print("boundary vert: ", nodes[row_start * n + i])
                res[curr_x, curr_y] = q
            elif i == col_start:
                print("starting corner: ", nodes[row_start * n + col_start])
                res[curr_x, curr_y] = calc_probs(q, "starting corner", curr_x, curr_y, res)
            elif i == col_end:
                print("top right corner: ", nodes[row_start * n + col_end])
                res[curr_x, curr_y] = calc_probs(q, "top right corner", curr_x, curr_y, res)
                print(res[curr_x, curr_y])
                break
            else:
                print("edge: ", nodes[row_start * n + i])
                res[curr_x, curr_y] = calc_probs(q, "top edge", curr_x, curr_y, res)
            print(res[curr_x, curr_y])
        row_start += 1

        for i in range(row_start, row_end + 1):
            curr_x = nodes[i * n + col_end][0]
            curr_y = nodes[i * n + col_end][1]
            if col_end == n-1:
                print("boundary vert: ", nodes[i * n + col_end])
                res[curr_x, curr_y] = q
            elif i == row_end:
                print("bottom right corner: ", nodes[row_end * n + col_end])
                res[curr_x, curr_y] = calc_probs(q, "bottom right corner", curr_x, curr_y, res)
                print(res[curr_x, curr_y])
                break
            else:
                print("edge: ", nodes[i * n + col_end])
                res[curr_x, curr_y] = calc_probs(q, "right edge", curr_x, curr_y, res)
            print(res[curr_x, curr_y])
        col_end -= 1

        if row_start <= row_end:
            for i in range(col_end, col_start - 1, -1):
                curr_x = nodes[row_end * n + i][0]
                curr_y = nodes[row_end * n + i][1]
                if row_end == n-1:
                    print("boundary vert: ", nodes[row_end * n + i])
                    res[curr_x, curr_y] = q
                elif i == col_start:
                    print("bottom left corner: ", nodes[row_end * n + col_start])
                    res[curr_x, curr_y] = calc_probs(q, "bottom left corner", curr_x, curr_y, res)
                    print(res[curr_x, curr_y])
                    break
                else:
                    print("edge: ", nodes[row_end * n + i])
                    res[curr_x, curr_y] = calc_probs(q, "bottom edge", curr_x, curr_y, res)
                print(res[curr_x, curr_y])
        row_end -= 1

        if col_start <= col_end:
            for i in range(row_end, row_start - 1, -1):
                curr_x = nodes[i * n + col_start][0]
                curr_y = nodes[i * n + col_start][1]
                if col_start == 0 and i == 1:
                    print("first agg: ", nodes[i * n + col_start])
                    res[curr_x, curr_y] = calc_probs(q, "first agg", curr_x, curr_y, res)
                elif col_start == 0:
                    print("boundary vert: ", nodes[i * n + col_start])
                    res[curr_x, curr_y] = q
                elif i == row_start:
                    print("end vertex: ", nodes[i * n + col_start])
                    res[curr_x, curr_y] = calc_probs(q, "end vertex", curr_x, curr_y, res)
                    ev += 1
                else:
                    print("edge: ", nodes[i * n + col_start])
                    res[curr_x, curr_y] = calc_probs(q, "left edge", curr_x, curr_y, res)
                print(res[curr_x, curr_y])
        col_start += 1


    res[row_start, col_end] = calc_probs(q, "final vert", row_start, col_end, res)
    print(ev)

    return res, res[row_start, col_end]

def calc_double(p1, p2, q):
    return p1 * p2 + (1 - p1) * p2 * q + (1 - p2) * p1 * q

def calc_triple(p1, p2, p3, q):
    return p1 * p2 * p3 + p1 * p2 * (1 - p3) * q + p1 * (1 - p2) * p3 * q + p1 * (1 - p2) * (1 - p3) * q + \
        (1 - p1) * p2 * p3 * q + (1 - p1) * p2 * (1 - p3) * q + (1 - p1) * (1 - p2) * p3 * q

def calc_quad(p1, p2, p3, p4, q):
    return (p1 * p2 * p3 * p4) + (p1 * p2 * p3 * (1 - p4)) + (p1 * p2 * (1 - p3) * p4) + (p1 * p2 * (1 - p3) * \
        (1 - p4) * q) + p1 * (1 - p2) * p3 * p4 + p1 * (1 - p2) * p3 * (1 - p4) * q + p1 * (1 - p2) * (1 - p3) * p4 \
        * q + (1 - p1) * p2 * p3 * p4 + (1 - p1) * p2 * p3 * (1 - p4) * q + (1 - p1) * p2 * (1 - p3) * p4 * q + \
        (1 - p1) * (1 - p2) * p3 * p4 * q

def calc_probs(q, spot, x, y, grid):

    if spot == "starting corner" or spot == "top edge":
        p1, p2 = grid[x, y - 1], grid[x - 1, y]
        return calc_double(p1, p2, q)
    elif spot == "first agg":
        p1, p2 = grid[x + 1, y], grid[x - 1, y]
        return calc_double(p1, p2, q)
    elif spot == "top right corner":
        p1, p2, p3 = grid[x, y - 1], grid[x - 1, y], grid[x, y + 1]
        return calc_triple(p1, p2, p3, q)
    elif spot == "bottom right corner":
        p1, p2, p3 = grid[x - 1, y], grid[x, y + 1], grid[x + 1, y]
        return calc_triple(p1, p2, p3, q)
    elif spot == "bottom left corner":
        p1, p2, p3 = grid[x, y + 1], grid[x + 1, y], grid[x, y - 1]
        return calc_triple(p1, p2, p3, q)
    elif spot == "right edge":
        p1, p2 = grid[x - 1, y], grid[x, y + 1]
        return calc_double(p1, p2, q)
    elif spot == "bottom edge":
        p1, p2 = grid[x + 1, y], grid[x, y + 1]
        return calc_double(p1, p2, q)
    elif spot == "left edge":
        p1, p2 = grid[x + 1, y], grid[x, y - 1]
        return calc_double(p1, p2, q)
    elif spot == "end vertex":
        p1, p2, p3 = grid[x-1, y], grid[x+1, y], grid[x, y-1]
        return calc_triple(p1, p2, p3, q)
    elif spot == "boundary":
        return q
    elif spot == "final vert":
        p1, p2, p3, p4 = grid[x - 1, y], grid[x + 1, y], grid[x, y - 1], grid[x, y + 1]
        return calc_quad(p1, p2, p3, p4, q)


def learning_rate_sample(grid):
    res = []
    col_idx = int(len(grid[0]) / 2)
    for row_idx in range(0, int(len(grid) / 2)):
        res.append(grid[row_idx, col_idx])

    return res

# a = 31
# grid = nx.grid_2d_graph(a, a)
# res, final = spiral_ordering_expectation(list(grid.nodes()), a, q)
# print("Expected learning rate:", np.sum(res) / a ** 2)
# print("Upper bound on expected: ", final)
# sample = learning_rate_sample(res)
# diff = np.diff(sample)
# diff = np.concatenate(([sample[0]], diff))
# print("samples: ", sample)
# print("diff: ", diff)
# print()
# sim_n_times("Basic Majority Model", 500, run_sim_2, (30**2, p, q, vg, vb, theta, grid, 30))
#
# layers = np.arange(len(sample))
# plt.figure(figsize=(8, 8)).subplots_adjust(hspace=0.6)
#
# plt.subplot(1, 2, 1)
# plt.plot(layers, sample, color="red")
# plt.xlabel("layer in grid")
# plt.ylabel("learning rate")
# plt.title("sample learning rate in each layer of grid")
#
# plt.subplot(1, 2, 2)
# plt.plot(layers, diff, color="orange")
# plt.xlabel("layer in grid")
# plt.ylabel("diff in learning rate")
# plt.title("diff in sample learning rate between each layer of grid")
# plt.show()

def spiral_ord_lb(q, n):
    if n % 2 == 0:
        return (4 * n - 4) * q + (np.floor((n - 2)/2)) * (3 * q ** 2 - 2 * q ** 3) + (np.floor((n - 2)/2) - 1) * \
            (4 * q ** 5 - 10 * q ** 4 + 6 * q ** 3 + q ** 2) + (np.floor((n - 2)/2) - 1)  * ()
    else:
        return

def rand_ord_lb(q, n):
    if n % 2 == 0:
        return (0.7 * 0.5) + ((2 * n - 2) * calc_triple(q, q, q, q) / n ** 2) + \
                (((n-2) ** 2 / 2) * calc_quad(q, q, q, q, q) / n ** 2)
    else:
        return ((0.7 * ((n ** 2) / 2 + 1)) + ((2 * n - 2) * calc_triple(q, q, q, q)) +
                ((n-2) ** 2 / 2) * calc_quad(q, q, q, q, q)) / n ** 2

# for i in range(5, 1000):
#     print(f"n={i}: {rand_ord_lb(0.51, i)}")
# print()

# sim_n_times("Basic Majority Model", 1000, run_sim, (100, p, q, vg, vb, theta, grid))
# print(f"n=10000: {rand_ord_lb(0.7, 10000)}")

# basic_majority_res = sim_n_times("Basic Majority Model", 500, run_sim,
#                                  (n, p, q, vg, vb, theta, power_law)
# aggregation_res = sim_n_times("Aggregation Model", 500, run_sim_aggregate,
#                              (n, p, q, vg, vb, theta, power_law, 0.001, 2))

#PA + arrival order = herding???
def run_sim_2(n, p, q, vg, vb, theta, network):
    #initialize network params
    init_net(n, q, network)

    #create a random node ordering
    ordering = random_ordering(list(network.nodes()))

    #construct the model
    basic_majority_model = BasicMajority(theta, q, p, vg, vb, network)
    basic_majority_model.make_decisions(ordering)

    #final measurements
    return basic_majority_model.calc_success_rate(), basic_majority_model.get_indep_decisions()

def calc_different_ns(k, m, num_trials, ns, sim_type, name):

    res_reg = np.zeros(len(ns))

    for i, nk in enumerate(ns):
        temp = np.zeros(num_trials)

        for j in range(1, num_trials):
            seed_clique = nx.complete_graph(k)
            run_sim(k, p, q, vg, vb, theta, seed_clique)
            power_law_2 = nx.barabasi_albert_graph(nk, m, seed=None, initial_graph=seed_clique)
            temp[j] = sim_type(nk, p, q, vg, vb, theta, power_law_2)[0]


        res_reg[i] = np.mean(temp)

    print(res_reg)
    print(f"Avg learning rate: {np.mean(res_reg)}")
    print(f"Learning rate range: [{np.min(res_reg)}, {np.max(res_reg)}]")
    print(f"Median: {np.median(res_reg)}")
    print(f"Variance: {np.var(res_reg)}")

    plt.plot(ns, res_reg, color="red", label=name)
    plt.xlabel("# of nodes")
    plt.ylabel("learning rate")
    plt.title(f"{name} avg learning rate")
    plt.show()

def calc_mult_diff_ns(k, m, num_trials, ns, sim_type_1, sim_type_2, name_1, name_2):
    res_1 = np.zeros(len(ns))
    res_2 = np.zeros(len(ns))

    for i, nk in enumerate(ns):
        temp_1 = np.zeros(num_trials)
        temp_2 = np.zeros(num_trials)

        for j in range(1, num_trials):
            seed_clique = nx.complete_graph(k)
            run_sim(k, p, q, vg, vb, theta, seed_clique)
            power_law_2 = nx.barabasi_albert_graph(nk, m, seed=None, initial_graph=seed_clique)
            temp_1[j] = sim_type_1(nk, p, q, vg, vb, theta, power_law_2)[0]
            temp_2[j] = sim_type_2(nk, p, q, vg, vb, theta, power_law_2)[0]

        res_1[i] = np.mean(temp_1)
        res_2[i] = np.mean(temp_2)

    print(f"{name_1} vs. {name_2}")
    print()
    print(f"Avg learning rate ({name_1}): {np.mean(res_1)}")
    print(f"Avg learning rate ({name_2}): {np.mean(res_2)}")
    print()
    print(f"Learning rate range ({name_1}): [{np.min(res_1)}, {np.max(res_1)}]")
    print(f"Learning rate range ({name_2}): [{np.min(res_2)}, {np.max(res_2)}]")
    print()
    print(f"Median ({name_1}): {np.median(res_1)}")
    print(f"Median ({name_2}): {np.median(res_2)}")
    print()
    print(f"Variance ({name_1}): {np.var(res_1)}")
    print(f"Variance ({name_2}): {np.var(res_2)}")

def calc_mult_diff_qs(k, m, num_trials, ns, qs, sim_type_1, sim_type_2, name_1, name_2):
    res_1 = np.zeros(len(qs))
    res_2 = np.zeros(len(qs))

    for i, _q in enumerate(qs):
        temp_1 = np.zeros(num_trials)
        temp_2 = np.zeros(num_trials)

        for j in range(1, num_trials):
            seed_clique = nx.complete_graph(k)
            run_sim(k, p, _q, vg, vb, theta, seed_clique)
            power_law_2 = nx.barabasi_albert_graph(ns, m, seed=None, initial_graph=seed_clique)
            temp_1[j] = sim_type_1(ns, p, _q, vg, vb, theta, power_law_2)[0]
            temp_2[j] = sim_type_2(ns, p, _q, vg, vb, theta, power_law_2)[0]

        res_1[i] = np.mean(temp_1)
        res_2[i] = np.mean(temp_2)

    plt.plot(qs, res_1, color="red", label=name_1)
    plt.plot(qs, res_2, color="orange", label=name_2)
    plt.xlabel("private signal")
    plt.ylabel("learning rate")
    plt.title(f"{name_1} vs. {name_2} avg learning rate")
    plt.legend()
    plt.show()

#calc_mult_diff_ns(20, 7, 200, np.arange(50, 501, 10), run_sim, run_sim_2, "arrival ordering", "random ordering")
#calc_different_ns(20, m, 200, np.arange(50, 501, 50), run_sim, "arrival ordering")
#calc_mult_diff_qs(20, 7, 200, 500, [0.51, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                  #run_sim, run_sim_2, "arrival ordering", "random ordering")

def run_sim3(n, p, q, vg, vb, theta, network, k):
    # initialize network params
    init_net(n, q, network)

    # create a random node ordering
    ordering = random_ordering(list(network.nodes()))

    # construct the model
    basic_aggregation_model = BasicAggregation(theta, q, p, vg, vb, network, k)
    basic_aggregation_model.make_decisions(ordering)

    # final measurements
    return basic_aggregation_model.calc_success_rate(), basic_aggregation_model.get_indep_decisions()

# ns = np.arange(50, 1001, 10)
# res = np.zeros(len(ns))

# for i, n in enumerate(ns):
#     p_law1 = nx.barabasi_albert_graph(n, k, seed=None, initial_graph=clique)
#     print("n: ", n)
#     res[i] = np.mean(sim_n_times("Basic Agg model", 500, run_sim3, (n, p, q, vg, vb, theta, p_law1, 6)))
#
# plt.plot(ns, res, color="red")
# plt.xlabel("# of nodes")
# plt.ylabel("learning rate")
# plt.title(f"basic aggregation avg learning rate")
# plt.show()

def run_sim_4(n, p, q, vg, vb, theta, network, high_value):
    # initialize network params
    init_net(n, q, network)

    # create a random node ordering
    ordering = random_ordering(list(network.nodes()))

    # construct the model
    two_neighbors_model = TwoNeighbors(theta, q, p, vg, vb, network, high_value)
    two_neighbors_model.make_decisions(ordering)

    # final measurements
    return two_neighbors_model.calc_success_rate(), two_neighbors_model.get_indep_decisions()

def run_sim_5(n, p, q, vg, vb, theta, network, threshold, k):
    # initialize network params
    init_net(n, q, network)

    # create a random node ordering
    ordering = degree_ordering_decreasing(network)

    # construct the model
    two_neighbors_model = TwoNeighborsHighVal(theta, q, p, vg, vb, network, threshold, k)
    two_neighbors_model.make_decisions(ordering)

    # final measurements
    return two_neighbors_model.calc_success_rate(), two_neighbors_model.get_indep_decisions()

#sim_n_times("Two Neighbors Model", 300, run_sim_4, (n, p, q, vg, vb, theta, erdos))

#ns = np.arange(50, 3001, 50)
# qs = np.arange(0.51, 0.99, 0.05)
# res_order = np.zeros(len(qs))
# res_random = np.zeros(len(qs))
#
new_n = 1000
#
# for i, q in enumerate(qs):
#     erdos = nx.erdos_renyi_graph(new_n, np.log(new_n)/new_n, seed=None, directed=False)
#     print("q: ", q)
#     res_order[i] = np.mean(sim_n_times("Two Neighbors Model", 300, run_sim_4, (new_n, p, q, vg, vb, theta, erdos)))
#     res_random[i] = np.mean(sim_n_times("Random ordering", 300, run_sim, (new_n, p, q, vg, vb, theta, erdos)))
#
# plt.plot(qs, res_order, color="red", label="two neighbors")
# plt.plot(qs, res_random, color="blue", label="random")
# plt.xlabel("q")
# plt.ylabel("learning rate")
# plt.title(f"q vs two neighbors model learning rate")
# plt.legend()
# plt.show()

ps = np.arange(0.005, 0.01, 0.002)
res_order = np.zeros(len(ps))
res_order_2 = np.zeros(len(ps))
res_random = np.zeros(len(ps))

for i, prob in enumerate(ps):
    erdos = nx.erdos_renyi_graph(new_n, prob, seed=None, directed=False)
    print("edge prob: ", prob)
    res_order[i] = np.mean(sim_n_times("Two Neighbors Model", 300, run_sim_4, (new_n, p, 0.7, vg, vb, theta, erdos, False)))
    res_order_2[i] = np.mean(sim_n_times("Two Neighbors + High Val", 300, run_sim_5, (new_n, p, 0.7, vg, vb, theta, erdos, 0.5, 8)))
    res_random[i] = np.mean(sim_n_times("Random ordering", 300, run_sim, (new_n, p, 0.7, vg, vb, theta, erdos)))

plt.plot(ps, res_order, color="red", label="two neighbors")
plt.plot(ps, res_order_2, color="orange", label="high value")
plt.plot(ps, res_random, color="blue", label="random")
plt.xlabel("edge prob")
plt.ylabel("learning rate")
plt.title(f"edge prob vs two neighbors model learning rate")
plt.legend()
plt.show()