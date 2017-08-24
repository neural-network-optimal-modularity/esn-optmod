import utilities
import graphgen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy.stats as stats
import scipy.linalg as linalg

def generate_network(N, mu, k, maxk, minc, maxc,
    deg_exp, weight_scale, lower_weight_bound,
    upper_weight_bound, temp_dir_ID, full_path):

    return graphgen.uniform_weighted_directed_lfr_graph(
        N, mu, k, maxk, minc, 
        maxc, weight_bounds=weight_scale * np.array(
            [lower_weight_bound, upper_weight_bound]),
        deg_exp=deg_exp, temp_dir_ID=temp_dir_ID, 
        full_path=full_path)

def adj_spectrum(network):

    return np.sort(nx.adjacency_spectrum(network))

def strength_matrix(network, axis=0):
    """
    Axis is the axis the sum will be taken over, e.g. rows or columns.
    Whether this corresponds to in or out strength depends upon notation.
    """

    A = np.asarray(nx.to_numpy_matrix(network))
    D = np.zeros(size=A.shape)
    for i in range(len(network)):
        D[i,i] = np.sum(A, axis=axis)

    return D

def norm_laplacian_spectrum(network, axis=0):
    
    A = np.asarray(nx.to_numpy_matrix(network))
    D = strength_matrix(network, axis)

    L = D - A
    if not np.allclose( np.dot(D, np.linalg.inv(D)), np.identity(size=D.shape) ):
        print("Problem with invertiding D")

    norm_L = np.dot(np.dot(np.linalg.inv(linalg.sqrtm(D)), L), np.linalg.inv(linalg.sqrtm(D)))
    return np.sort(np.eigvals(norm_L))

def spectral_analysis(network_parameters, savefile=True):

    largest_eigvals_adj_by_mu = []
    second_largest_eigvals_adj_by_mu = []
    # second_smallest_eigvals_lap_by_mu = []
    spectrums = {'adj_eigvals': [], 'mus': network_parameters['mus']}
    for mu in network_parameters['mus']:

        adj_eigvals_by_trial = []
        # lap_eigvals_by_trial = []

        largest_eigvals_adj_by_trial = []
        second_largest_eigvals_adj_by_trial = []
        # second_smallest_eigvals_lap_by_trial = []
        for j in range(network_parameters['num_reservoir_samplings']):
            network = generate_network(N=network_parameters['N'], 
                mu=mu, 
                k=network_parameters['k'], 
                maxk=network_parameters['maxk'], 
                minc=network_parameters['minc'], 
                maxc=network_parameters['maxc'], 
                deg_exp=network_parameters['deg_exp'], 
                temp_dir_ID=network_parameters['temp_dir_ID'], 
                full_path=network_parameters['full_path'], 
                weight_scale=network_parameters['reservoir_weight_scale'],
                lower_weight_bound=network_parameters['lower_reservoir_bound'], 
                upper_weight_bound=network_parameters['upper_reservoir_bound'])

            adj_eigvals = adj_spectrum(network)
            # lap_eigvals = norm_laplacian_spectrum(network)

            largest_eigvals_adj_by_trial.append(adj_eigvals[-1])
            second_largest_eigvals_adj_by_trial.append(adj_eigvals[-2])
            # second_smallest_eigvals_lap_by_trial.append(lap_eigvals[1])

            adj_eigvals_by_trial.append(adj_eigvals)
            # lap_eigvals_by_trial.append(lap_eigvals)

        largest_eigvals_adj_by_mu.append(largest_eigvals_adj_by_trial)
        second_largest_eigvals_adj_by_mu.append(second_largest_eigvals_adj_by_trial)

        spectrums['adj_eigvals'].append(adj_eigvals_by_trial)
        # spectrums['lap_eigvals'].append(lap_eigvals_by_trial)

    utilities.save_object(spectrums, network_parameters['command_prefix'] + "_spectrum.pyobj")

    return np.array(largest_eigvals_adj_by_mu), \
        np.array(second_largest_eigvals_adj_by_mu)

def plot_spectrum_versus_mu(mu, eigvals, ylabel, prefix):

    plt.clf()
    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in eigvals:
        list_means.append(np.mean(trials))
        sem = stats.sem(trials)
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    plt.errorbar(mu, list_means, yerr=[list_CIlow, list_CIhigh],
        marker='o', ls='-', color='blue')
    plt.xlabel(r"$\mu$")
    plt.ylabel(ylabel)
    plt.savefig(prefix + "_spectrum.png", dpi=300)
    plt.close()
    plt.clf()

def plot_combined_spectral_versus_mu(mu, first_eigens, second_eigens, prefix):

    fig, ax1 = plt.subplots()

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in first_eigens:
        list_means.append(np.mean(trials))
        sem = stats.sem(trials)
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    ax1.errorbar(mu, list_means, yerr=[list_CIlow, list_CIhigh],
        marker='o', ls='-', color='black', label='largest')

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in second_eigens:
        list_means.append(np.mean(trials))
        sem = stats.sem(trials)
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    ax1.errorbar(mu, list_means, yerr=[list_CIlow, list_CIhigh],
        marker='o', ls='-', color='blue', label='second')

    ax2 = ax1.twinx()

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in np.fabs(first_eigens) - np.fabs(second_eigens):
        list_means.append(np.mean(trials))
        sem = stats.sem(trials)
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    ax2.errorbar(mu, list_means, yerr=[list_CIlow, list_CIhigh],
        marker='o', ls='--', color='red', label='gap')

    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel("eigenvalue")
    ax2.set_ylabel("gap", color='r')
    ax2.tick_params('y', colors='r')
    ax1.legend(loc="upper center")
    fig.tight_layout()
    plt.savefig(prefix + "_combined_spectrum.png", dpi=300)
    plt.close()
    plt.clf()

if __name__ == '__main__':
    """
    """
    params = {'backend': 'ps', 
              'axes.labelsize': 20, 
              'text.fontsize': 20, 
              'legend.fontsize': 14, 
              'xtick.labelsize': 15, 
              'ytick.labelsize': 15, 
              'text.usetex': True, 
              'xtick.major.size': 10, 
              'ytick.major.size': 10, 
              'xtick.minor.size': 8, 
              'ytick.minor.size': 8, 
              'xtick.major.width': 1.0, 
              'ytick.major.width': 1.0, 
              'xtick.minor.width': 1.0,
              'ytick.minor.width': 1.0,
              'pdf.fonttype' : 42,
              'ps.fonttype' : 42}
    plt.rcParams.update(params)

    network_parameters = {
        'num_reservoir_samplings': 128,#128,
        'N': 500,
        'mus': np.linspace(0.0, 0.5, 20),
        'k': 6,
        'maxk': 6,
        'minc':10,
        'maxc':10,
        'deg_exp':1.0,
        'lower_reservoir_bound': -0.2,
        'upper_reservoir_bound': 1.0,
        'reservoir_weight_scale': 1.132,
        'temp_dir_ID':0, 
        'full_path': '/home/nathaniel/workspace/function_of_modularity/',
        'command_prefix': 'MC_N500_ws1.132_com10'
    }

    largest_eigvals_adj, second_largest_eigvals_adj = spectral_analysis(network_parameters)
    plot_spectrum_versus_mu(network_parameters['mus'], 
        largest_eigvals_adj, "largest adj", "largest_adj")
    plot_spectrum_versus_mu(network_parameters['mus'], 
        second_largest_eigvals_adj, "second largest adj", "second_largest_adj")
    plot_spectrum_versus_mu(network_parameters['mus'], 
        np.fabs(largest_eigvals_adj) - np.fabs(second_largest_eigvals_adj), "spectral gap", "spectral_gap")
    plot_combined_spectral_versus_mu(network_parameters['mus'], 
        largest_eigvals_adj, second_largest_eigvals_adj, "spectral_gap")
    # plot_spectrum_versus_mu(network_parameters['mus'], 
    #     second_smallest_eigvals_lap, "second smallest laplacian", "second_smallest_laplacian")