# from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import networkx as nx
matplotlib.use('Agg')
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import CircuitUtilities as cu
from multiprocessing import Pool
from functools import partial
import numpy as np
import random as rnd
import DESN_modeler
import TwoCommunityBlockModel
import scipy.stats as stats

NUM_PROCESS = 22
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

    net_FPs = []
    community_FPs = []
    for trial in xrange(input_dict['num_trials']):
        graph = TwoCommunityBlockModel.two_com_graph_uniform(input_dict['num_neurons'], 
            input_dict['mu'], input_dict['avg_degree'], input_dict['res_weight_bounds'][0], 
            input_dict['res_weight_bounds'][1])
        DESNmodel = DESN_modeler.DiscreteEchoStateModel(graph, seed=rnd.randint(1,32000), **(input_dict['esn_pars']))
        DESNmodel.Run()
        
        net_FP = DESNmodel.last_activity()

        community_FP = []
        for community in DESNmodel.GetCommunityNodeDict(input_dict['esn_pars']['community_key']).keys():
            community_FP.append(DESNmodel.community_last_activity(community, input_dict['esn_pars']['community_key']))

        net_FPs.append(net_FP)
        community_FPs.append(community_FP)

    return [ input_dict['mu'], input_dict['coupling_str'], input_dict['Ao'], \
        input_dict['esn_pars']['signal_ratio'], np.mean(net_FP) ] + list(np.mean(community_FPs, axis=0))

def signal_ratio_to_Ao_map(sig_sequence, input_weight, net_size, sigmoid_pars):
    """
    Maps a set of signal ratios to the IC activity of the network. Assumes homogeneous signal weights/str
    given as input_weight. Sigmoid parameters and network size must also be given.
    """

    input_neuron_activation = sigmoid(input_weight, **sigmoid_pars)
    hidden_neuron_activation = sigmoid(0.0, **sigmoid_pars)
    return sig_sequence * input_neuron_activation * net_size + hidden_neuron_activation * (1. - sig_sequence) * net_size

def sigmoid(x, a=1.0, b=1.0, c=0.0, d=0.0, e=1.0):
    """
    numpy Vector/matrix friendly sigmoid function
    """
    return a / (b + np.exp(-e*(x-c))) + d

def Ao_to_signal_ratio_map(Ao_sequence, input_weight, net_size, sigmoid_pars):
    """
    Maps a set of IC of activity to an input sequence. Assumes homogeneous signal weights/strength
    given as input_weight. sigmoid parameters and network size must also be given.
    """

    input_neuron_activation = sigmoid(input_weight, **sigmoid_pars)
    hidden_neuron_activation = sigmoid(0.0, **sigmoid_pars)

    return ((np.array(Ao_sequence) / net_size) - hidden_neuron_activation) / (input_neuron_activation - hidden_neuron_activation)

def stable_FP_coordinates(listSig_results, error=0.01, com=0):
    """
    """

    sorted_listSig_results = sorted(listSig_results, key=lambda x: x[2])
    # find fixed points within (error):
    list_mu = [ sorted_listSig_results[0][0] ]
    list_coupling = [ sorted_listSig_results[0][1] ]
    list_FP = [ sorted_listSig_results[0][4+com] ]
    for i in xrange(len(sorted_listSig_results) - 1):
        if abs(sorted_listSig_results[i][4+com] - sorted_listSig_results[i+1][4+com]) > error:
            list_mu.append(sorted_listSig_results[i+1][0])
            list_coupling.append(sorted_listSig_results[i+1][1])
            list_FP.append(sorted_listSig_results[i+1][4+com])

    return list_mu, list_coupling, list_FP

def merge_stableFP(listMus_listCoupling_listSig_results, error=0.01, com=0):
    """
    Returns three arrays: one for mu, coupling, and A (of FP)
    each are of the same length and represent a point in space
    """

    mu_coordinates = []
    coupling_coordinates = []
    FP_coordinates = []
    for i, listCoupling_listSig_results in enumerate(listMus_listCoupling_listSig_results):
        for j, listSig_results in enumerate(listCoupling_listSig_results):
            mu, coupling, FP = stable_FP_coordinates(listSig_results, error, com)
            mu_coordinates += mu 
            coupling_coordinates += coupling
            FP_coordinates += FP

    return np.array(mu_coordinates), np.array(coupling_coordinates), np.array(FP_coordinates)

def unstable_FP_coordinates(listSig_results, error=0.01, com=0):
    """
    """

    sorted_listSig_results = sorted(listSig_results, key=lambda x: x[2])
    # find fixed points within (error):
    list_mu = [ ]
    list_coupling = [ ]
    list_FP = [ ]
    for i in xrange(len(sorted_listSig_results) - 1):
        if abs(sorted_listSig_results[i][4+com] - sorted_listSig_results[i+1][4+com]) > error:
            list_mu.append(sorted_listSig_results[i+1][0])
            list_coupling.append(sorted_listSig_results[i+1][1])
            list_FP.append( (sorted_listSig_results[i+1][2] + sorted_listSig_results[i][2]) / 2.)

    return list_mu, list_coupling, list_FP

def find_unstableFP(listMus_listCoupling_listSig_results, error=0.01, com=0):
    """
    Returns three arrays: one for mu, coupling, and A (of FP)
    each are of the same length and represent a point in space
    """

    mu_coordinates = []
    coupling_coordinates = []
    FP_coordinates = []
    for i, listCoupling_listSig_results in enumerate(listMus_listCoupling_listSig_results):
        for j, listSig_results in enumerate(listCoupling_listSig_results):
            mu, coupling, FP = unstable_FP_coordinates(listSig_results, error, com)
            mu_coordinates += mu 
            coupling_coordinates += coupling 
            FP_coordinates += FP

    return np.array(mu_coordinates), np.array(coupling_coordinates), np.array(FP_coordinates)    

def equilibrium_surface_plot(prefix, listMus_listCoupling_listSig_results, FP_error=1.0, com=0):
    """
    """

    # Merge FPs [get one or more stable FP per mu/coupling value] - returns x,y,z pos of each one
    stable_mu, stable_coupling, stable_FP = merge_stableFP(listMus_listCoupling_listSig_results, FP_error, com)
    # Find unstable FPs /need to consider possibility of multiple unstable FP??? walk through take diff pos+1 - pos 
    unstable_mu, unstable_coupling, unstable_FP = find_unstableFP(listMus_listCoupling_listSig_results, FP_error, com)

    # Plot stable FP (blue)
    fig=plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter3D(stable_mu, stable_coupling, stable_FP, color='blue', edgecolors='none')
    # ax.plot3D(stable_mu, stable_coupling, stable_FP, color='blue')

    # Plot unstable FP (red)
    ax.scatter3D(unstable_mu, unstable_coupling, unstable_FP, color='red', edgecolors='none')
    # ax.plot3D(unstable_mu, unstable_coupling, unstable_FP, color='red')

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$w_s$')
    ax.set_zlabel(r'$A$')

    # ax.azim
    # ax.elev

    for ii in xrange(0,360,30):
        ax.view_init(elev=10., azim=ii)
        plt.savefig(prefix + "_equilibrium_surface_plot_" + str(ii) + "_com" + str(com) + ".png", dpi=300)

    plt.clf()
    plt.close()

if __name__ == '__main__':

    params = {'backend': 'ps', \
              'axes.labelsize': 20, \
              'font.size': 20, \
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

    prefix = "test"
    list_mus = np.linspace(0.005, 0.1, 10)
    list_signal_ratio = np.linspace(0.01, 0.2, 10)
    list_Ao = signal_ratio_to_Ao_map(list_signal_ratio, input_weight=40.0, 
        net_size=500, sigmoid_pars={'c':1, 'e':10})
    list_coupling_str = np.linspace(0.1, 1.5, 10)
    parallel_input_sequence = [ {'esn_pars':{
        'niter': 20000, 
        'input_series': np.array(np.ones((1,1,1))),
        'input_weight_scale': 40.0,
        'neuron_type':'sigmoid',
        'neuron_pars': {'c':1, 'e':10},
        'initialization_state':(0.0,0.0),
        'input_weight_bounds':(1.0, 1.0),
        'signal_ratio': signal_ratio,
        'community_key': "community",
        'target_community': 1 },
        'mu': mu,
        'Ao': list_Ao[j],
        'coupling_str': coupling_str,
        'num_neurons': 500, 
        'avg_degree': 7,  
        'num_trials': 10,
        'res_weight_bounds': coupling_str * np.array((1.00, 1.00))}
            for mu in list_mus 
            for coupling_str in list_coupling_str
            for j, signal_ratio in enumerate(list_signal_ratio) ]

    # Parallelize function
    SimulateCurcuit.parallel = parallel_function(SimulateCurcuit)
    model_results = SimulateCurcuit.parallel(parallel_input_sequence)

    # Re-order output
    num_mus = len(list_mus)
    num_signals = len(list_signal_ratio)
    num_coupling = len(list_coupling_str)
    listMus_listCoupling_listSig_results = [ [ [model_results[i*num_signals*num_coupling + j*num_signals + k] \
        for k in xrange(num_signals) ] for j in xrange(num_coupling) ] for i in xrange(num_mus)]

    # Save
    cu.save_object((list_mus, list_signal_ratio, list_coupling_str, listMus_listCoupling_listSig_results), \
        prefix + "_simresults_equilibrium_surface.pyobj")

    # Load
    # list_mus, list_signal_ratio, list_coupling_str, listMus_listCoupling_listSig_results = \
    #     cu.load_object('test_simresults_equilibrium_surface.pyobj')

    # Make equilibrium surface plots
    equilibrium_surface_plot(prefix, listMus_listCoupling_listSig_results, 10.0, 0)
    equilibrium_surface_plot(prefix, listMus_listCoupling_listSig_results, 10.0, 1)
    equilibrium_surface_plot(prefix, listMus_listCoupling_listSig_results, 10.0, 2)