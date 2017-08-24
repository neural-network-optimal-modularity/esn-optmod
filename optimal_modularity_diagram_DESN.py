""" A script written to generate publication graphs of the optimal modularity phase diagram for artificial neuronal networks. This script runs in parallel in order to build the multiple trials for each point and many number of points required to construct the phase diagram. It can also plot chosen slices through that diagram. """

import matplotlib
import networkx as nx
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import utilities
from multiprocessing import Pool
from functools import partial
import numpy as np
import random as rnd
import DESN_modeler
import graphgen
import scipy.stats as stats

NUM_PROCESS = 28
def parallel_function(f):
    def easy_parallize(f, sequence):
        """ assumes f takes sequence as input, easy w/ Python's scope """
        pool = Pool(processes=NUM_PROCESS) # depends on available cores
        result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
        cleaned = [x for x in result if not x is None] # getting results
        cleaned = np.asarray(cleaned)
        pool.close() # not optimal! but easy
        pool.join()
        return cleaned
    
    return partial(easy_parallize, f)

def SimulateCurcuit(input_dict):
    # Reset seeds for parallel processing
    rnd.seed()
    np.random.seed()

    graph = graphgen.uniform_weighted_two_community_graph(input_dict['num_neurons'], input_dict['mu'], input_dict['avg_degree'], input_dict['res_weight_bounds'][0], input_dict['res_weight_bounds'][1])
    if input_dict['spectral_radius'] != None:
        adjust_graph_spectral_radius(graph, input_dict['spectral_radius'])
    DESNmodel = DESN_modeler.DiscreteEchoStateModel(graph, signal_ratio=input_dict['signal_ratio'], 
        seed=rnd.randint(1,32000),**(input_dict['esn_pars']))
    DESNmodel.Run()
    
    net_activity = DESNmodel.net_activity()
    net_activation_time = DESNmodel.net_activation_time(input_dict['activation_threshold'])
    net_halting_time = DESNmodel.net_halting_time(input_dict['deactivation_threshold'])
    net_FP = DESNmodel.last_activity()

    community_activity = []
    community_activation_times = []
    community_halting_times = []
    community_FP = []
    for community in DESNmodel.GetCommunityNodeDict(input_dict['community_key']).keys():
        community_activity.append(DESNmodel.community_activity(community, input_dict['community_key']))
        community_activation_times.append(DESNmodel.community_activation_time(community, input_dict['community_key'], input_dict['activation_threshold']))
        community_halting_times.append(DESNmodel.community_halting_time(community, input_dict['community_key'], input_dict['deactivation_threshold']))
        community_FP.append(DESNmodel.community_last_activity(community, input_dict['community_key']))

    return [ input_dict['mu'], input_dict['signal_ratio'], net_activity , net_activity ] + community_activity + [net_activation_time] \
            + community_activation_times + [net_halting_time] + community_halting_times + [net_FP] + community_FP

def adjust_graph_spectral_radius(graph, new_spectral_radius):

    largest_eigenval = max(np.abs(np.real(np.linalg.eigvals(nx.adjacency_matrix(graph).A))))
    for edge in graph.edges_iter():
        graph[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight'] / largest_eigenval * new_spectral_radius

def halting_time_contour_plot(prefix, list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, levels=None, xscale=None, yscale=None):

    # Uses halting time - index=9
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[9]

    plt.clf()
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label(r"$\textup{time} \: (ms)$")
    plt.xlabel("$\mu$", fontsize=30)
    plt.ylabel("$r_{sig}$", fontsize=30)
    if xscale == 'log':
        plt.xscale('log')
    if yscale == 'log':
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_halting-time_contour.png')
    plt.clf()
    plt.close()

def halting_time_vs_mu_plot(prefix, ratio_index, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults):
    # Uses community halting times
    listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    arrayMus_arrayTrials_com1haltingtime = arrayMus_arrayTrials_arrayResults[:,:,10]
    arrayMus_arrayTrials_com2haltingtime = arrayMus_arrayTrials_arrayResults[:,:,11]

    list_com1means = []
    list_com1CIhigh = []
    list_com1CIlow = []
    for trials in arrayMus_arrayTrials_com1haltingtime:
        com1mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com1means.append(com1mean[0])
        list_com1CIhigh.append(com1mean[1][1] - com1mean[0])
        list_com1CIlow.append(com1mean[0] - com1mean[1][0])

    list_com2means = []
    list_com2CIhigh = []
    list_com2CIlow = []
    for trials in arrayMus_arrayTrials_com2haltingtime:
        com2mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com2means.append(com2mean[0])
        list_com2CIhigh.append(com2mean[1][1] - com2mean[0])
        list_com2CIlow.append(com2mean[0] - com2mean[1][0])

    signal_ratio = list_signal_ratio[ratio_index]

    plt.errorbar(list_mus, list_com1means, yerr=[list_com1CIlow, list_com1CIhigh], marker='o', ls='-', color='blue', label='Seed community')
    plt.errorbar(list_mus, list_com2means, yerr=[list_com2CIlow, list_com2CIhigh], marker='o', ls='-', color='red', label='Neighboring Community')
    plt.ylabel('Halting time ($steps$)')
    plt.xlabel('$\mu$')
    plt.gca().set_ylim(bottom=0)
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    plt.legend(loc="upper right",frameon=False)
    plt.tight_layout()
    plt.savefig(prefix + "_" + str(round(signal_ratio,3)) + "_halting-time_vs_modularity.png", dpi=300)
    plt.close()
    plt.clf()

def activity_contour_plot(prefix, list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, levels=None, xscale=None, yscale=None):
    # Uses total activity index=3
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[3]

    plt.clf()
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label(r"Activity")
    plt.xlabel("$\mu$", fontsize=30)
    plt.ylabel("$r_{sig}$", fontsize=30)
    if xscale == 'log':
        plt.xscale('log')
    if yscale == 'log':
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_activity_contour.png')
    plt.clf()
    plt.close()

def activity_vs_mu_plot(prefix, ratio_index, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults):
    # Uses community activity
    listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    arrayMus_arrayTrials_com1activity = arrayMus_arrayTrials_arrayResults[:,:,4]
    arrayMus_arrayTrials_com2activity = arrayMus_arrayTrials_arrayResults[:,:,5]

    list_com1means = []
    list_com1CIhigh = []
    list_com1CIlow = []
    for trials in arrayMus_arrayTrials_com1activity:
        com1mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com1means.append(com1mean[0])
        list_com1CIhigh.append(com1mean[1][1] - com1mean[0])
        list_com1CIlow.append(com1mean[0] - com1mean[1][0])

    list_com2means = []
    list_com2CIhigh = []
    list_com2CIlow = []
    for trials in arrayMus_arrayTrials_com2activity:
        com2mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com2means.append(com2mean[0])
        list_com2CIhigh.append(com2mean[1][1] - com2mean[0])
        list_com2CIlow.append(com2mean[0] - com2mean[1][0])

    signal_ratio = list_signal_ratio[ratio_index]

    plt.errorbar(list_mus, list_com1means, yerr=[list_com1CIlow, list_com1CIhigh], marker='o', ls='-', color='blue', label='Seed community')
    plt.errorbar(list_mus, list_com2means, yerr=[list_com2CIlow, list_com2CIhigh], marker='o', ls='-', color='red', label='Neighboring Community')
    plt.gca().set_ylim(bottom=0)
    plt.ylabel('Activity')
    plt.xlabel('$\mu$')
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    plt.legend(loc="upper right",frameon=False)
    plt.tight_layout()
    plt.savefig(prefix + "_" + str(round(signal_ratio,3)) + "_activity_vs_modularity.png", dpi=300)
    plt.close()
    plt.clf()

def fixed_point_contour_plot(prefix, list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, levels=None, xscale=None, yscale=None):
    # Uses net FP index=12
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[12] /500.# divide by cirtuit size

    plt.clf()
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label(r"Activity")
    plt.xlabel("$\mu$", fontsize=30)
    plt.ylabel("$r_{sig}$", fontsize=30)
    if xscale == 'log':
        plt.xscale('log')
    if yscale == 'log':
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_FP_contour.png')
    plt.clf()
    plt.close()

def fixed_point_vs_mu_plot(prefix, ratio_index, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults):
    # Uses community FP
    listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    arrayMus_arrayTrials_com1activity = arrayMus_arrayTrials_arrayResults[:,:,13]
    arrayMus_arrayTrials_com2activity = arrayMus_arrayTrials_arrayResults[:,:,14]

    list_com1means = []
    list_com1CIhigh = []
    list_com1CIlow = []
    for trials in arrayMus_arrayTrials_com1activity:
        com1mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com1means.append(com1mean[0])
        list_com1CIhigh.append(com1mean[1][1] - com1mean[0])
        list_com1CIlow.append(com1mean[0] - com1mean[1][0])

    list_com2means = []
    list_com2CIhigh = []
    list_com2CIlow = []
    for trials in arrayMus_arrayTrials_com2activity:
        com2mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com2means.append(com2mean[0])
        list_com2CIhigh.append(com2mean[1][1] - com2mean[0])
        list_com2CIlow.append(com2mean[0] - com2mean[1][0])

    signal_ratio = list_signal_ratio[ratio_index]

    plt.errorbar(list_mus, list_com1means, yerr=[list_com1CIlow, list_com1CIhigh], marker='o', ls='-', color='blue', label='Seed community')
    plt.errorbar(list_mus, list_com2means, yerr=[list_com2CIlow, list_com2CIhigh], marker='o', ls='-', color='red', label='Neighboring Community')
    plt.gca().set_ylim(bottom=0)
    plt.ylabel('Activity')
    plt.xlabel('$\mu$')
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    plt.legend(loc="upper right",frameon=False)
    plt.tight_layout()
    plt.savefig(prefix + "_" + str(round(signal_ratio,3)) + "_FP_vs_modularity.png", dpi=300)
    plt.close()
    plt.clf()

def time_to_activation_contour_plot(prefix, list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, levels=None, xscale=None, yscale=None):
    # Uses burst time to activation index=6
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[6]

    plt.clf()
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label(r"$\textup{time} \: (steps)$")
    plt.xlabel("$\mu$", fontsize=30)
    plt.ylabel("$r_{sig}$", fontsize=30)
    if xscale == 'log':
        plt.xscale('log')
    if yscale == 'log':
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_time-to-activation_contour.png')
    plt.clf()
    plt.close()

def time_to_activation_vs_mu_plot(prefix, ratio_index, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults):
    # Uses community activation times
    listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    arrayMus_arrayTrials_com1activationtime = arrayMus_arrayTrials_arrayResults[:,:,7]
    arrayMus_arrayTrials_com2activationtime = arrayMus_arrayTrials_arrayResults[:,:,8]

    list_com1means = []
    list_com1CIhigh = []
    list_com1CIlow = []
    for trials in arrayMus_arrayTrials_com1activationtime:
        com1mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com1means.append(com1mean[0])
        list_com1CIhigh.append(com1mean[1][1] - com1mean[0])
        list_com1CIlow.append(com1mean[0] - com1mean[1][0])

    list_com2means = []
    list_com2CIhigh = []
    list_com2CIlow = []
    for trials in arrayMus_arrayTrials_com2activationtime:
        com2mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_com2means.append(com2mean[0])
        list_com2CIhigh.append(com2mean[1][1] - com2mean[0])
        list_com2CIlow.append(com2mean[0] - com2mean[1][0])

    signal_ratio = list_signal_ratio[ratio_index]

    plt.errorbar(list_mus, list_com1means, yerr=[list_com1CIlow, list_com1CIhigh], marker='o', ls='-', color='blue', label='Seed community')
    plt.errorbar(list_mus, list_com2means, yerr=[list_com2CIlow, list_com2CIhigh], marker='o', ls='-', color='red', label='Neighboring Community')
    plt.ylabel('Time to activation ($steps$)')
    plt.xlabel('$\mu$')
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    plt.gca().set_ylim(bottom=0)
    plt.legend(loc="upper right",frameon=False)
    plt.tight_layout()
    plt.savefig(prefix + "_" + str(round(signal_ratio,3)) + "_time-to-activation_vs_modularity.png", dpi=300)
    plt.close()
    plt.clf()

def bayes_mvs_wrapper(sequence, alpha):

    if max(sequence) == min(sequence):
        return (np.mean(sequence), (np.mean(sequence), np.mean(sequence))), (0, (0,0)), (0, (0,0))
    else:

        return stats.bayes_mvs(sequence, alpha)

if __name__ == '__main__':

    params = {'backend': 'ps', \
              'axes.labelsize': 20, \
              'text.fontsize': 20, \
              'legend.fontsize': 16, \
              'xtick.labelsize': 20, \
              'ytick.labelsize': 20, \
              'text.usetex': True, \
              'xtick.major.size': 10, \
              'ytick.major.size': 10, \
              'xtick.minor.size': 8, \
              'ytick.minor.size': 8, \
              'xtick.major.width': 1.0, \
              'ytick.major.width': 1.0, \
              'xtick.minor.width': 1.0, \
              'ytick.minor.width': 1.0}
    plt.rcParams.update(params)

    # Example parameters: niter:100, np.ones((1,1,1)), input scale: 40.0, input bouds (1.0, 1.0), 
    # sigmoid at {'c':1, 'e':10}, deg=7, res_weights: (0.5, 0.5)

    # Circuit parameters
    prefix = "paper_version_rw0.4_iw3_is0.5_k6_trials100"
    esn_parameters = {"niter": 150, 'input_series':np.array(np.ones((150,1,1))), 'input_weight_scale': 0.5, 'neuron_type':'sigmoid', 
        'neuron_pars': {'c':1, 'e':10}, 'initialization_state':(0.0,0.0), 'input_weight_bounds':(3.0, 3.0),
        'target_community': 1, 'community_key': 'community'}
    num_neurons = 500
    spectral_radius = None
    avg_degree = 6
    list_mus = np.linspace(0.0, 0.5, 20) #0.005, 0.5, 20
    list_signal_ratio = np.linspace(0.01, 0.2, 20)
    num_trials = 100
    activation_threshold = 10.
    deactivation_threshold = 0.001
    parallel_input_sequence = [ {'mu':mu, 'signal_ratio': signal_ratio, 'community_key': 'community',
        'num_neurons': num_neurons, 'avg_degree': avg_degree, 'esn_pars':esn_parameters,  
        'activation_threshold': activation_threshold, 'deactivation_threshold': deactivation_threshold, 
        'spectral_radius': spectral_radius, 'res_weight_bounds': (0.4, 0.4)} \
            for mu in list_mus \
            for signal_ratio in list_signal_ratio \
            for trial in xrange(num_trials) ]

    # Parallelize function
    SimulateCurcuit.parallel = parallel_function(SimulateCurcuit)
    model_results = SimulateCurcuit.parallel(parallel_input_sequence)

    # Re-order output
    num_mus = len(list_mus)
    num_ratios = len(list_signal_ratio)
    listRatios_listMus_listTrials_listResults = [ [ [ model_results[(num_trials*num_ratios*i + num_trials*j + k)] \
        for k in xrange(num_trials)] \
        for i in xrange(num_mus)] \
        for j in xrange(num_ratios) ]
    listRatios_listMus_listMeanResults = [ [ np.mean([ model_results[(num_trials*num_ratios*i + num_trials*j + k)] \
        for k in xrange(num_trials)], axis=0) \
        for i in xrange(num_mus)] \
        for j in xrange(num_ratios) ]

    # Sort
    sorted_listRatios_listMus_listMeanResults = sorted([ sorted(inner_list, key=lambda x: x[0]) 
        for inner_list in listRatios_listMus_listMeanResults ], key=lambda x: x[0][1])
    utilities.save_object((sorted(list_mus), sorted(list_signal_ratio), 
        sorted_listRatios_listMus_listMeanResults, 
        listRatios_listMus_listTrials_listResults), 
        prefix + "_simresults_vs_mu_vs_signal.pyobj")

    # Plot results
    # list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults, \
    #     listRatios_listMus_listTrials_listResults = utilities.load_object("test0.4_simresults_vs_mu_vs_signal.pyobj")
    # activity_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)
    # activity_vs_mu_plot(prefix, 9, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)
    fixed_point_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)
    # fixed_point_vs_mu_plot(prefix, 5, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)
    # halting_time_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)
    # time_to_activation_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)
    # halting_time_vs_mu_plot(prefix, 9, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)
    # time_to_activation_vs_mu_plot(prefix, 0, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)