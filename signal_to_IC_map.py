"""
Creates a plot showing how the input signal influences (r_o) the initial condition (A_o) [with constant signal_weight]
"""

import matplotlib.pyplot as plt 
import numpy as np 
import DESN_modeler
import graphgen
import sys

def map_model(r_sequence, num_input_options, max_neuron_exite=1.0):
    return np.array(r_sequence) * max_neuron_exite * num_input_options

def signal_ratio_to_Ao_map(sig_sequence, input_weight, net_size, sig_pars):
    """
    """

    input_neuron_activation = sigmoid(input_weight, **sig_pars)
    hidden_neuron_activation = sigmoid(0.0, **sig_pars)
    return sig_sequence * input_neuron_activation * net_size + hidden_neuron_activation * (1. - sig_sequence) * net_size

def sigmoid(x, a=1.0, b=1.0, c=0.0, d=0.0, e=1.0):
    """
    numpy Vector/matrix friendly sigmoid function
    """
    return a / (b + np.exp(-e*(x-c))) + d


def Ao_to_signal_ratio_map(Ao_sequence, input_weight, net_size, sig_pars):
    """
    """

    input_neuron_activation = sigmoid(input_weight, **sig_pars)
    hidden_neuron_activation = sigmoid(0.0, **sig_pars)

    return ((np.array(Ao_sequence) / net_size) - hidden_neuron_activation) / (input_neuron_activation - hidden_neuron_activation)

if __name__ == '__main__':

    # Plot r_sig vs A_o
    tmax = 1
    N = 500
    mu = 0.1
    avgD = 7
    signal_weights = [0.5, 1.0, 40.0]
    signal_ratios = np.linspace(0.0, 0.25, 15)
    for signal_weight in signal_weights:
        listA_o = []
        for signal_ratio in signal_ratios:
            graph = graphgen.uniform_weighted_two_community_graph(N, mu, avgD, 0.50, 0.50)
            DESM = DESN_modeler.DiscreteEchoStateModel(graph, signal_weight, signal_ratio, tmax, np.ones((1,1,1)), neuron_type='sigmoid', 
                neuron_pars={'e':10, 'c':1}, initialization_state=(0.0,0.0), input_weight_bounds=(1.0,1.0))
            DESM.Run(1, 'community', 1)
            listA_o.append(DESM.net_activity())

        plt.plot(signal_ratios, listA_o, marker='o', ls='none', label=str(round(signal_weight, 2)))
        # plt.plot(signal_ratios, signal_ratio_to_Ao_map(signal_ratios, signal_weight, 500, {'e':10, 'c':1}), ls='-')
        plt.plot(Ao_to_signal_ratio_map(listA_o, signal_weight, 500, {'e':10, 'c':1}), listA_o, ls='-')
    plt.xlabel(r'$r_{sig}$')
    plt.ylabel(r'$A_o$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal_to_IC_map.png", dpi=300)

    # # Plot A_o vs signal_strength
    # tmax = 1
    # N = 500
    # mu = 0.1
    # avgD = 7
    # graph = graphgen.uniform_weighted_two_community_graph(N, mu, avgD, 0.50, 0.50)
    # signal_weights = np.linspace(0.01, 5., 100)
    # signal_ratios = [0.04, 0.08, 0.16, 0.32] 
    # for signal_ratio in signal_ratios:
    #     listA_o = []
    #     for signal_weight in signal_weights:
    #         DESM = DESN_modeler.DiscreteEchoStateModel(graph, signal_weight, signal_ratio, tmax, np.ones((1,1,1)), neuron_type='sigmoid', 
    #             neuron_pars={'e':10, 'c':1}, initialization_state=(0.0,0.0), input_weight_bounds=(1.0,1.0))
    #         DESM.Run(1, 'community', 1)
    #         listA_o.append(DESM.net_activity())

    #     plt.plot(signal_weights, listA_o, lw=2, label=r"$r_{sig}=$ " + str(round(signal_ratio, 2)))
    # plt.xlabel(r'$w_{sig}$')
    # plt.ylabel(r'$A_o$')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("sigweight_to_IC_map.png", dpi=300)

    # Total signal strength... # nodes * amplitude