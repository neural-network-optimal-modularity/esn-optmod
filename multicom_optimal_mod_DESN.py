""" A script written to generate publication graphs of the optimal modularity phase diagram 
for artificial neuronal networks. This script runs in parallel in order to build the 
multiple trials for each point and many number of points required to construct the 
phase diagram. It can also plot chosen slices through that diagram. """

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

NUM_PROCESS = 30
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

    print('\nworker_id:', input_dict['worker_id'], '\n')
    graph = graphgen.uniform_weighted_directed_lfr_graph_asnx(N=input_dict['num_neurons'], 
        mu=input_dict['mu'],  k=input_dict['avg_degree'], maxk=input_dict['avg_degree'], 
        maxc=input_dict['maxc'], minc=input_dict['minc'], weight_bounds=input_dict['res_weight_bounds'],
        temp_dir_ID=input_dict['temp_dir_ID'], full_path=input_dict['full_path'])
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
    for community in DESNmodel.GetCommunityNodeDict(input_dict['esn_pars']['community_key']).keys():
        community_activity.append(DESNmodel.community_activity(community, input_dict['esn_pars']['community_key']))
        community_activation_times.append(DESNmodel.community_activation_time(community, input_dict['esn_pars']['community_key'], input_dict['activation_threshold']))
        community_halting_times.append(DESNmodel.community_halting_time(community, input_dict['esn_pars']['community_key'], input_dict['deactivation_threshold']))
        community_FP.append(DESNmodel.community_last_activity(community, input_dict['esn_pars']['community_key']))

    return [ input_dict['mu'], input_dict['signal_ratio'], 
            len(DESNmodel.GetCommunityNodeDict(input_dict['esn_pars']['community_key']).keys()),
            net_activity ] + \
            community_activity + [net_activation_time] + \
            community_activation_times + [net_halting_time] + community_halting_times + [net_FP] + community_FP

def adjust_graph_spectral_radius(graph, new_spectral_radius):

    largest_eigenval = max(np.abs(np.real(np.linalg.eigvals(nx.adjacency_matrix(graph).A))))
    for edge in graph.edges_iter():
        graph[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight'] / largest_eigenval * new_spectral_radius

def halting_time_contour_plot(prefix, list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, levels=None, xscale=None, yscale=None):

    # Uses halting time - index=9
    num_com = int(listRatios_listMus_listMeanResults[0][0][2])
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[4+num_com+1+num_com]

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

    num_com = int(listRatios_listMus_listTrials_listResults[0][0][0][2])
    for com_index in range(num_com):
        # Uses community halting times
        listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
        arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
        arrayMus_arrayTrials_comhaltingtime = arrayMus_arrayTrials_arrayResults[:,:,4+num_com+1+num_com+1+com_index]

        list_commeans = []
        list_comCIhigh = []
        list_comCIlow = []
        for trials in arrayMus_arrayTrials_comhaltingtime:
            commean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
            list_commeans.append(commean[0])
            list_comCIhigh.append(commean[1][1] - commean[0])
            list_comCIlow.append(commean[0] - commean[1][0])

        plt.errorbar(list_mus, list_commeans, yerr=[list_comCIlow, list_comCIhigh], marker='None', ls='-')

    signal_ratio = list_signal_ratio[ratio_index]
    plt.ylabel('Halting time ($steps$)')
    plt.xlabel('$\mu$')
    plt.gca().set_ylim(bottom=0)
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    # plt.legend(loc="upper right",frameon=False)
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

    num_com = int(listRatios_listMus_listTrials_listResults[0][0][0][2])
    for com_index in range(num_com):
        # Uses community activity
        listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
        arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
        arrayMus_arrayTrials_comactivity = arrayMus_arrayTrials_arrayResults[:,:,4+com_index]

        list_commeans = []
        list_comCIhigh = []
        list_comCIlow = []
        for trials in arrayMus_arrayTrials_comactivity:
            commean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
            list_commeans.append(commean[0])
            list_comCIhigh.append(commean[1][1] - commean[0])
            list_comCIlow.append(commean[0] - commean[1][0])
        plt.errorbar(list_mus, list_commeans, yerr=[list_comCIlow, list_comCIhigh], marker='None', ls='-')

    signal_ratio = list_signal_ratio[ratio_index]
    plt.gca().set_ylim(bottom=0)
    plt.ylabel('Activity')
    plt.xlabel('$\mu$')
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    # plt.legend(loc="upper right",frameon=False)
    plt.tight_layout()
    plt.savefig(prefix + "_" + str(round(signal_ratio,3)) + "_activity_vs_modularity.png", dpi=300)
    plt.close()
    plt.clf()

def fixed_point_contour_plot(prefix, list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, levels=None, xscale=None, yscale=None):
    # Uses net FP index=
    num_com = int(listRatios_listMus_listMeanResults[0][0][2])
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[4+num_com+1+num_com+1+num_com]/500. # divide by size of network

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

    num_com = int(listRatios_listMus_listTrials_listResults[0][0][0][2])
    for com_index in range(num_com):   
        # Uses community FP
        listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
        arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
        arrayMus_arrayTrials_comactivity = arrayMus_arrayTrials_arrayResults[:,:,4+num_com+1+num_com+1+num_com+1+com_index]

        list_commeans = []
        list_comCIhigh = []
        list_comCIlow = []
        for trials in arrayMus_arrayTrials_comactivity:
            commean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
            list_commeans.append(commean[0])
            list_comCIhigh.append(commean[1][1] - commean[0])
            list_comCIlow.append(commean[0] - commean[1][0])

        plt.errorbar(list_mus, list_commeans, yerr=[list_comCIlow, list_comCIhigh], marker='None', ls='-')

    signal_ratio = list_signal_ratio[ratio_index]
    plt.gca().set_ylim(bottom=0)
    plt.ylabel('Activity')
    plt.xlabel('$\mu$')
    plt.title("Signal fraction ($r_{sig}$): " + str(round(signal_ratio,3)))
    plt.legend(loc="upper right",frameon=False)
    plt.tight_layout()
    plt.savefig(prefix + "_" + str(round(signal_ratio,3)) + "_FP_vs_modularity.png", dpi=300)
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
    prefix = "N500_randinput_trials100_same_as_MC_task"#"paper_50com_example_no_target_com_trials100"
    # esn_parameters = {"niter": 1000, 'input_series':np.random.randint(2, size=(1000, 1, 1)), 'input_weight_scale': 1., 
    #     'neuron_type':'sigmoid', 
    #     'neuron_pars': {'c':1, 'e':10}, 'initialization_state':(0.0,0.0), 'input_weight_bounds':(-0.2, 1.0),
    #     'target_community': None, 'community_key': 'community'} # [1,2,3,4,5,6,7,8,9,10]
    # num_neurons = 500
    # spectral_radius = None
    # avg_degree = 6
    # list_mus = np.linspace(0.0, 0.5, 20) #0.005, 0.5, 20
    # list_signal_ratio = np.linspace(0.1, 0.6, 20) # [0.25789474,  0.33684211, 0.41578947,  0.49473684]
    # num_trials = 100#100
    # num_mus = len(list_mus)
    # num_ratios = len(list_signal_ratio)
    # activation_threshold = 10.
    # deactivation_threshold = 0.001
    # parallel_input_sequence = [ {'mu':mu, 'signal_ratio': signal_ratio, 
    #     'num_neurons': num_neurons, 'avg_degree': avg_degree, 'esn_pars':esn_parameters, 
    #     'minc': 10, 'maxc': 10, 'temp_dir_ID': prefix + str(num_trials*num_ratios*i + num_trials*j + k), 
    #     'full_path':'/nobackup/njrodrig/topology_of_function/',#"/home/nathaniel/workspace/function_of_modularity/",
    #     'activation_threshold': activation_threshold, 'deactivation_threshold': deactivation_threshold, 
    #     'spectral_radius': spectral_radius, 'res_weight_bounds': (1.132 * -0.2, 1.132 * 1.00),
    #     'worker_id': num_trials*num_ratios*i + num_trials*j + k} \
    #         for i, mu in enumerate(list_mus) \
    #         for j, signal_ratio in enumerate(list_signal_ratio) \
    #         for k, trial in enumerate(range(num_trials)) ]

    # # Parallelize function
    # SimulateCurcuit.parallel = parallel_function(SimulateCurcuit)
    # model_results = SimulateCurcuit.parallel(parallel_input_sequence)

    # # Re-order output
    # listRatios_listMus_listTrials_listResults = [ [ [ model_results[(num_trials*num_ratios*i + num_trials*j + k)] \
    #     for k in xrange(num_trials)] \
    #     for i in xrange(num_mus)] \
    #     for j in xrange(num_ratios) ]
    # listRatios_listMus_listMeanResults = [ [ np.mean([ model_results[(num_trials*num_ratios*i + num_trials*j + k)] \
    #     for k in xrange(num_trials)], axis=0) \
    #     for i in xrange(num_mus)] \
    #     for j in xrange(num_ratios) ]

    # # Sort
    # sorted_listRatios_listMus_listMeanResults = sorted([ sorted(inner_list, key=lambda x: x[0]) for inner_list in listRatios_listMus_listMeanResults ], key=lambda x: x[0][1])
    # utilities.save_object((sorted(list_mus), sorted(list_signal_ratio), sorted_listRatios_listMus_listMeanResults, listRatios_listMus_listTrials_listResults), prefix + "_simresults_vs_mu_vs_signal.pyobj")

    # Plot results
    list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults, \
        listRatios_listMus_listTrials_listResults = utilities.load_object(prefix + "_simresults_vs_mu_vs_signal.pyobj")
    activity_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)

    activity_vs_mu_plot(prefix, 6, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)
    # fixed_point_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults, levels=np.linspace(0,1,11))
    # fixed_point_vs_mu_plot(prefix, 3, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)
    # halting_time_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)
    # halting_time_vs_mu_plot(prefix, 3, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)
    # time_to_activation_contour_plot(prefix, list_mus, list_signal_ratio, sorted_listRatios_listMus_listMeanResults)
    # time_to_activation_vs_mu_plot(prefix, 0, list_mus, list_signal_ratio, listRatios_listMus_listTrials_listResults)