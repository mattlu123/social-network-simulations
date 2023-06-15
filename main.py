import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

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

#make list of high degree nodes
def get_top_percentile_nodes(graph, percentile):
    # Get the degrees of all nodes in the graph
    degrees = dict(graph.degree())

    # Calculate the threshold degree for the given percentile
    sorted_degrees = sorted(degrees.values(), reverse=True)
    threshold_degree = sorted_degrees[int(len(sorted_degrees) * percentile / 100)]

    # Find the nodes with degrees above the threshold
    top_percentile_nodes = [node for node, degree in degrees.items() if degree >= threshold_degree]

    return top_percentile_nodes

def get_lowest_percentile_neighbors(graph, node, percentile):
    # Get the degrees of all neighbors of the node
    degrees = dict(graph.degree(graph.neighbors(node)))

    # Calculate the threshold degree for the given percentile
    sorted_degrees = sorted(degrees.values())
    threshold_degree = sorted_degrees[int(len(sorted_degrees) * percentile / 100)]

    # Find the neighbors with degrees below the threshold
    lowest_percentile_neighbors = [n for n, degree in degrees.items() if degree <= threshold_degree]

    return lowest_percentile_neighbors

def makeDecision(n, p, q,vg, vb, theta, network, agent):
        
    signal = nx.get_node_attributes(network, "private signal")[agent]
    neighbors = [n for n in network.neighbors(agent)]
    actions = nx.get_node_attributes(network, "action")
    n_actions = [actions[key] for key in neighbors]
    choice = np.random.choice([theta, 1 - theta], p=[q, 1 - q])

    zeros = [num for num in n_actions if num == 0]
    ones = [num for num in n_actions if num == 1]
    
    if len(neighbors) < 2 or (len(n_actions) - n_actions.count(-1)) < 2:
        #payoff = vg * ((p*signal)/(p*signal + (1-p)*(1-signal))) + vb * ((1-p)*(1-signal)/(p*signal + (1-p)*(1-signal)))
        network.nodes[agent]["action"] = choice 
    elif len(zeros) - len(ones) > 1:
        network.nodes[agent]["action"] = 0
    elif len(ones) - len(zeros) > 1:
        network.nodes[agent]["action"] = 1
    else:
       # payoff = vg * ((p * signal) / (p * signal + (1 - p) * (1 - signal))) + vb * (
        #            (1 - p) * (1 - signal) / (p * signal + (1 - p) * (1 - signal)))
        network.nodes[agent]["action"] = choice 

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
    #ordering = np.random.permutation(np.arange(0,n))
    
    for agent in ordering:
        makeDecision(n, p, q,vg, vb, theta, network, agent)

    #final measurements
    final_actions = np.array(list(nx.get_node_attributes(network, "action").values()))
    success_rate = len(final_actions[final_actions == theta])/len(final_actions)
    return success_rate

#set node information
def run_sim_aggregate(n, p, q, vg, vb, theta, network):
    index = np.around(np.linspace(0, n-1, n)).astype(int)
    priv_signal = np.full(n, q)
    priv_signals = dict(zip(index, priv_signal))
    action = np.full(n, -1)
    actions = dict(zip(index, action))
    nx.set_node_attributes(network, priv_signals, "private signal")
    nx.set_node_attributes(network, actions, "action")

    #create a random node ordering
    ordering = np.arange(n-1,-1,-1)
    #ordering = np.random.permutation(np.arange(0,n))
   
    #aggregate step
    highDegreeNodes = get_top_percentile_nodes(network, 5)
    for agent in highDegreeNodes:
        lowDegreeNeighbors = get_lowest_percentile_neighbors(network, agent, 5)
        
        for LowAgents in lowDegreeNeighbors:
            if network.nodes[LowAgents]["action"] == -1:
                makeDecision(n,p,q,vg,vb,theta,network,LowAgents)
            
        signal = nx.get_node_attributes(network, "private signal")[agent]
        neighbors = lowDegreeNeighbors
        actions = nx.get_node_attributes(network, "action")
        n_actions = [actions[key] for key in neighbors]
        choice = np.random.choice([theta, 1 - theta], p=[q, 1 - q])

        zeros = [num for num in n_actions if num == 0]
        ones = [num for num in n_actions if num == 1]
        
        if len(neighbors) < 2 or (len(n_actions) - n_actions.count(-1)) < 2:
            #payoff = vg * ((p*signal)/(p*signal + (1-p)*(1-signal))) + vb * ((1-p)*(1-signal)/(p*signal + (1-p)*(1-signal)))
            network.nodes[agent]["action"] = choice 
        elif len(zeros) - len(ones) > 1:
            network.nodes[agent]["action"] = 0
        elif len(ones) - len(zeros) > 1:
            network.nodes[agent]["action"] = 1
        else:
            # payoff = vg * ((p * signal) / (p * signal + (1 - p) * (1 - signal))) + vb * (
            #            (1 - p) * (1 - signal) / (p * signal + (1 - p) * (1 - signal)))
            network.nodes[agent]["action"] = choice 

    #run simulation
    for agent in ordering:
        if network.nodes[agent]["action"] == -1:
            makeDecision(n, p, q,vg, vb, theta, network, agent)

    #final measurements
    final_actions = np.array(list(nx.get_node_attributes(network, "action").values()))
    success_rate = len(final_actions[final_actions == theta])/len(final_actions)
    return success_rate

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
    total_runs[i] = run_sim_aggregate(n, p, q, vg, vb, theta, power_law)

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





