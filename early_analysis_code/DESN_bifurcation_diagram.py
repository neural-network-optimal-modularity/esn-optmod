import matplotlib
import networkx as nx
matplotlib.use('Agg')
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

def Ao_to_signal_ratio_map(Ao_sequence, input_weight, net_size, sigmoid_pars):
    """
    Maps a set of IC of activity to an input sequence. Assumes homogeneous signal weights/strength
    given as input_weight. sigmoid parameters and network size must also be given.
    """

    input_neuron_activation = sigmoid(input_weight, **sigmoid_pars)
    hidden_neuron_activation = sigmoid(0.0, **sigmoid_pars)

    return ((np.array(Ao_sequence) / net_size) - hidden_neuron_activation) / (input_neuron_activation - hidden_neuron_activation)

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

def stable_fp_coordinates(listSig_results, error=0.01, com=0):
    """
    """

    sorted_listSig_results = sorted(listSig_results, key=lambda x: x[4+com])
    # find fixed points within (error):
    list_coupling = [ sorted_listSig_results[0][1] ]
    list_FP = [ sorted_listSig_results[0][4+com] ]
    for i in xrange(len(sorted_listSig_results) - 1):
        if abs(sorted_listSig_results[i][4+com] - sorted_listSig_results[i+1][4+com]) > error:
            list_coupling.append(sorted_listSig_results[i+1][1])
            list_FP.append(sorted_listSig_results[i+1][4+com])

    return list_coupling, list_FP

def find_stable_fp(listCoupling_listSig_results, error=0.01, com=0):
    """
    """

    coupling_coordinates = []
    FP_coordinates = []
    for i, listSig_results in enumerate(listCoupling_listSig_results):
        coupling, FP = stable_fp_coordinates(listSig_results, error, com)
        coupling_coordinates += coupling
        FP_coordinates += FP

    return np.array(coupling_coordinates), np.array(FP_coordinates)

def unstable_fp_coordinates(listSig_results, error=0.01, com=0):
    """
    """

    sorted_listSig_results = sorted(listSig_results, key=lambda x: x[2])
    # find fixed points within (error):
    list_coupling = [ ]
    list_FP = [ ]
    for i in xrange(len(sorted_listSig_results) - 1):
        if abs(sorted_listSig_results[i][4+com] - sorted_listSig_results[i+1][4+com]) > error:
            list_coupling.append(sorted_listSig_results[i+1][1])
            list_FP.append( (sorted_listSig_results[i+1][2] + sorted_listSig_results[i][2]) / 2.)

    return list_coupling, list_FP

def find_unstable_fp(listCoupling_listSig_results, error=0.01, com=0):
    """
    Returns two arrays: one for coupling, and A (of FP)
    each are of the same length and represent a point in space
    """

    coupling_coordinates = []
    FP_coordinates = []
    for i, listSig_results in enumerate(listCoupling_listSig_results):
            coupling, FP = unstable_fp_coordinates(listSig_results, error, com)
            coupling_coordinates += coupling 
            FP_coordinates += FP

    return np.array(coupling_coordinates), np.array(FP_coordinates)    

def bifurcation_diagram(prefix, listCoupling_listSig_results, fp_error=1.0, com=0):
    """
    """

    stable_coupling, stable_fp = find_stable_fp(listCoupling_listSig_results, error=fp_error, com=com)
    unstable_coupling, unstable_fp = find_unstable_fp(listCoupling_listSig_results, error=fp_error, com=com)

    plt.clf()
    plt.scatter(stable_coupling, stable_fp, color='blue')
    plt.scatter(unstable_coupling, unstable_fp, color='red')

    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$A$')

    plt.savefig(prefix + "_bifucation_diagram_plot_com" + str(com) + ".png", dpi=300)
    plt.close()
    plt.clf()

def grid(x, y, z, resX=100, resY=100):
    """
    Convert 3 column data to matplotlib grid
    """
    from matplotlib.mlab import griddata

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def bifurcation_contour(prefix, listCoupling_listSig_results, com=0):
    """
    """

    coupling_values = [ results[1] for listSig_results in listCoupling_listSig_results for results in listSig_results ]
    activation_values = [ results[2] for listSig_results in listCoupling_listSig_results for results in listSig_results ]
    fp_values = [ results[4+com] for listSig_results in listCoupling_listSig_results for results in listSig_results ]
    coupling, activation, fp = grid(coupling_values, activation_values, fp_values)

    plt.clf()
    plt.contourf(coupling, activation, fp)

    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$A$')
    plt.colorbar()
    plt.savefig(prefix + "_bifucation_contour_com" + str(com) + ".png", dpi=300)
    plt.close()
    plt.clf()


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
    mu = 0.1
    list_signal_ratio = np.linspace(0.01, 0.5, 40)
    list_Ao = signal_ratio_to_Ao_map(list_signal_ratio, input_weight=40.0, 
        net_size=500, sigmoid_pars={'c':1, 'e':10})
    list_coupling_str = np.linspace(0.0, 1.2, 40)
    parallel_input_sequence = [ {'esn_pars':{
        'niter': 1000, 
        'input_series': np.array(np.ones((1,1,1))),
        'input_weight_scale': 40.0,
        'neuron_type':'sigmoid',
        'neuron_pars': {'c':1, 'e':10},
        'initialization_state':(0.0,0.0),
        'input_weight_bounds':(1.0, 1.0),
        'signal_ratio': signal_ratio,
        'target_community': 1, 
        'community_key': "community" },
        'mu': mu,
        'Ao': list_Ao[j],
        'coupling_str': coupling_str,
        'num_neurons': 500, 
        'avg_degree': 7,
        'num_trials': 1,
        'res_weight_bounds': coupling_str * np.array((1.00, 1.00))}
            for coupling_str in list_coupling_str
            for j, signal_ratio in enumerate(list_signal_ratio) ]

    # Parallelize function
    SimulateCurcuit.parallel = parallel_function(SimulateCurcuit)
    model_results = SimulateCurcuit.parallel(parallel_input_sequence)

    # Re-order output
    num_signals = len(list_signal_ratio)
    num_coupling = len(list_coupling_str)
    listCoupling_listSig_results = [ [model_results[j*num_signals + k] \
        for k in xrange(num_signals) ] for j in xrange(num_coupling) ]

    # Save
    cu.save_object((mu, list_signal_ratio, list_coupling_str, listCoupling_listSig_results), \
        prefix + "_simresults_bifurcation_diagram.pyobj")

    # # Load
    # mu, list_signal_ratio, list_coupling_str, listCoupling_listSig_results = \
    #     cu.load_object(prefix + '_simresults_bifurcation_diagram.pyobj')

    # Make bifurcation diagrams
    # bifurcation_diagram(prefix, listCoupling_listSig_results, 10.0, 0)
    # bifurcation_contour(prefix, listCoupling_listSig_results)