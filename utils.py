import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

def sim_n_times(name, num_trials, sim_func, other_params):
    total_runs = np.zeros(num_trials)
    total_indep_dec = np.zeros(num_trials)

    for i in range(0, num_trials):
        total_runs[i], total_indep_dec[i] = sim_func(*other_params)

    total_runs = np.array(total_runs)

    print(name)
    print(f"Avg learning rate: {np.mean(total_runs)}")
    print(f"Learning rate range: [{np.min(total_runs)}, {np.max(total_runs)}]")
    print(f"Cascades: {len(total_runs[total_runs < other_params[2]])}")
    print(f"Avg independent decisions: {np.mean(total_indep_dec)}")
    print()

    return total_runs

def plot_histograms(name, runs):
    #Plot the histogram
    plt.hist(runs, bins=100)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.show()

