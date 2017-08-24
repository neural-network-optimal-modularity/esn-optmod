"""
Used to check if a input graph meets the characteristics of a reference graph.
"""
import networkx as nx
import sys
import numpy as np 
import matplotlib.pyplot as plt
import CircuitUtilities as cu
import scipy.stats as stats

class SliceGraphChecker(object):

    def __init__(self, reference_graph, input_graph, weight_key="weight", \
        x_key="x", y_key="y", community_key="community", node_id="global_id", prefix='test'):
        """
        These must be networkx graphs with keys related to weight, position, community keys, and global node id.
        The nodes in each graph must have a counterpart: the global node id is used.
        Also, community IDs in each graph must correspond to each other.
        """

        self.reference_graph = reference_graph
        self.input_graph = input_graph
        self.weight_key = weight_key
        self.x_key = x_key
        self.y_key = y_key
        self.community_key = community_key
        self.prefix = prefix

    def CheckGlobalReciprocity(self):
        pass

    def CheckNetworkWeightDistribution(self, plot=False, xlog=True, ylog=True):
        """
        Does a KS-test on the edge weight distributions (optionally plots the CCDF of the two distributions)
        """

        # Get reference weights
        reference_weights = [ self.reference_graph[edge[0]][edge[1]][self.weight_key] for edge in self.reference_graph.edges_iter() ]

        # Get input weights
        input_weights = [ self.input_graph[edge[0]][edge[1]][self.weight_key] for edge in self.input_graph.edges_iter() ]

        # Perform KS-test
        Kstat, Pval = stats.ks_2samp(reference_weights, input_weights)

        # (optional plot)
        if plot:
            plt.clf()
            ref_x, ref_y = cu.eccdf(reference_weights)
            plt.plot(ref_x, ref_y, 'ko', label='Reference')
            in_x, in_y = cu.eccdf(input_weights)
            plt.plot(in_x, in_y, 'bo', label='Input')
            
            if xlog:
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
            plt.ylabel('CCDF')
            plt.xlabel('weight')
            plt.savefig(self.prefix + "_xlog" + str(xlog) + "_ylog" + str(ylog) + '_network_weight_comparison.png',dpi=300)
            plt.close()
            plt.clf()

        return Kstat, Pval

    def CheckNetworkDegreeDistribution(self, plot=False, direction='in', xlog=True, ylog=True):
        """
        Does a KS-test on the degree distribution (optionally plots the CCDF of the two distributions)
        """

        if direction == 'in':
            # Get reference degrees
            reference_degrees = self.reference_graph.in_degree(self.reference_graph.nodes(), weight=None).values()

            # Get intput degrees
            input_degrees = self.input_graph.in_degree(self.input_graph.nodes(), weight=None).values()

            filename = 'network_in-degree_comparison.png'

        else:
            # Get reference degrees
            reference_degrees = self.reference_graph.out_degree(self.reference_graph.nodes(), weight=None).values()

            # Get intput degrees
            input_degrees = self.input_graph.out_degree(self.input_graph.nodes(), weight=None).values()

            filename = 'network_out-degree_comparison.png'

        # Perform KS-test
        Kstat, Pval = stats.ks_2samp(reference_degrees, input_degrees)

        # (optional plot)
        if plot:
            plt.clf()
            ref_x, ref_y = cu.eccdf(reference_degrees)
            plt.plot(ref_x, ref_y, 'ko', label='Reference')
            in_x, in_y = cu.eccdf(input_degrees)
            plt.plot(in_x, in_y, 'bo', label='Input')
            
            if xlog:
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
            plt.ylabel('CCDF')
            plt.xlabel('degree')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(self.prefix + '_' + "_xlog" + str(xlog) + "_ylog" + str(ylog) + filename,dpi=300)
            plt.close()
            plt.clf()

        return Kstat, Pval

    def CheckNetworkNodeStrengthDistribution(self, plot=False, direction='in', xlog=True, ylog=True, slog=False):
        """
        Does a KS-test on the degree distribution (optionally plots the CCDF of the two distributions)
        """

        # Get reference strength
        if not slog:
            if direction == 'in':
                reference_strengths = self.reference_graph.in_degree(self.reference_graph.nodes(), weight='weight').values()
            else:
                reference_strengths = self.reference_graph.out_degree(self.reference_graph.nodes(), weight='weight').values()

            # Get intput strength
            if direction == 'in':
                input_strengths = self.input_graph.in_degree(self.input_graph.nodes(), weight='weight').values()
            else:
                input_strengths = self.input_graph.out_degree(self.input_graph.nodes(), weight='weight').values()
        else:

            if direction == 'in':
                reference_strengths = cu.node_strength_list(self.reference_graph, True)
            else:
                reference_strengths = cu.node_strength_list(self.reference_graph, False)

            # Get intput strength
            if direction == 'in':
                input_strengths = cu.node_strength_list(self.input_graph, True)
            else:
                input_strengths = cu.node_strength_list(self.input_graph, False)
            

        # Perform KS-test
        Kstat, Pval = stats.ks_2samp(reference_strengths, input_strengths)

        # (optional plot)
        if plot:
            plt.clf()
            ref_x, ref_y = cu.eccdf(reference_strengths)
            plt.plot(ref_x, ref_y, 'ko', label='Reference')
            in_x, in_y = cu.eccdf(input_strengths)
            plt.plot(in_x, in_y, 'bo', label='Input')
            
            if xlog:
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
            plt.ylabel('CCDF')
            plt.xlabel('strengths')
            plt.legend(loc='lower left')
            plt.tight_layout()
            if direction == 'in':
                filename = self.prefix + "_xlog" + str(xlog) + "_ylog" + str(ylog) + '_network_in-strengths_comparison.png'
            else:
                filename = self.prefix + "_xlog" + str(xlog) + "_ylog" + str(ylog) + '_network_out-strengths_comparison.png'
            plt.savefig(filename,dpi=300)
            plt.close()
            plt.clf()

        return Kstat, Pval

    def CheckDistanceToWeightRelationship(self, plot=False):
        pass

    def CheckNeuronInWeightDistribution(self, plot=False):
        """
        Runs a KS-test on each neuron in the network to determine if they are 
        satisfying reference graph.
        Optionally you can plot the distribution of p-values that result to see
        how many deviate (using a box-plot)
        """

        # Get in-weights for each neuron
        reference_dictNode_listWeights = cu.global_node_weight_dict(self.reference_graph, in_weight=True)
        input_dictNode_listWeights = cu.global_node_weight_dict(self.input_graph, in_weight=True)

        # Loop through each node in input and check
        Kstats = []
        Pvals = []
        for node in input_dictNode_listWeights.keys():
            ks, pval = stats.ks_2samp(reference_dictNode_listWeights[node], input_dictNode_listWeights[node])
            Kstats.append(ks)
            Pvals.append(pval)

        # Make box-plot of pvalues
        if plot:
            plt.clf()
            plt.boxplot(Pvals)
            plt.ylabel('p-value')
            plt.savefig(self.prefix + "_xlog" + str(xlog) + "_ylog" + str(ylog) + '_pvalues_of_neuron_in-weight_distribution_comparison.png', dpi=300)
            plt.close()
            plt.clf()

        return np.median(Kstats), np.median(Pvals)

    def CheckNeuronOutWeightDistribution(self, plot=False):
        """
        Runs a KS-test on each neuron in the network to determine if they are 
        satisfying reference graph.
        Optionally you can plot the distribution of p-values that result to see
        how many deviate (using a box-plot)
        """

        # Get in-weights for each neuron
        reference_dictNode_listWeights = cu.global_node_weight_dict(self.reference_graph, in_weight=False)
        input_dictNode_listWeights = cu.global_node_weight_dict(self.input_graph, in_weight=False)

        # Loop through each node in input and check
        Kstats = []
        Pvals = []
        for node in input_dictNode_listWeights.keys():
            ks, pval = stats.ks_2samp(reference_dictNode_listWeights[node], input_dictNode_listWeights[node])
            Kstats.append(ks)
            Pvals.append(pval)

        # Make box-plot of pvalues
        if plot:
            plt.clf()
            plt.boxplot(Pvals)
            plt.ylabel('p-value')
            plt.savefig(self.prefix + "_xlog" + str(xlog) + "_ylog" + str(ylog) + '_pvalues_of_neuron_out-weight_distribution_comparison.png', dpi=300)
            plt.close()
            plt.clf()

        return np.median(Kstats), np.median(Pvals)

    def CheckNeuronStrengthDifferences(self, plot=False, xlog=True, ylog=True):

        # Get in and out strengths
        reference_in_strengths = np.array(cu.node_strength_list(self.reference_graph, True))
        reference_out_strengths = np.array(cu.node_strength_list(self.reference_graph, False))

        input_in_strengths = np.array(cu.node_strength_list(self.input_graph, True))
        input_out_strengths = np.array(cu.node_strength_list(self.input_graph, False))

        # Calculate differences
        reference_diff = np.abs(reference_in_strengths - reference_out_strengths)
        input_diff = np.abs(input_in_strengths - input_out_strengths)

        # Perform KS-test
        Kstat, Pval = stats.ks_2samp(reference_diff, input_diff)

        # (optional plot)
        if plot:
            plt.clf()
            ref_x, ref_y = cu.eccdf(reference_diff)
            plt.plot(ref_x, ref_y, 'ko', label='Reference')
            in_x, in_y = cu.eccdf(input_diff)
            plt.plot(in_x, in_y, 'bo', label='Input')
            
            if xlog:
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
            plt.ylabel('CCDF')
            plt.xlabel('in/out difference')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(self.prefix + "_xlog" + str(xlog) + "_ylog" + str(ylog) + "_network_strength_difference_comparison.png",dpi=300)
            plt.close()
            plt.clf()

        return Kstat, Pval

    def CheckCommunityDifferences(self):
        """
        Calculates community comparison quantities
        """

        # Get information about edges from each graph
        ref_com_info = cu.graph_edge_information(self.reference_graph, self.community_key)
        input_com_info = cu.graph_edge_information(self.input_graph, self.community_key)

        # Fractional difference between total weights (close to zero when same, - when input is larger, + when ref is larger)
        total_weight_diff = (ref_com_info['total_weight'] - input_com_info['total_weight']) / (input_com_info['total_weight'] + ref_com_info['total_weight'])

        # Difference between intracommunity weight (only meaningufl when total_weight_diff is small)
        intracommunity_weight_diff = ref_com_info['intracommunity_weight'] - input_com_info['intracommunity_weight']


        # Get community weight differences
        ref_com_weight_diff = cu.community_weight_fraction(self.reference_graph, self.community_key)
        input_com_weight_diff = cu.community_weight_fraction(self.input_graph, self.community_key)
    
        return total_weight_diff, intracommunity_weight_diff, ref_com_info['intercommunity_weight'], input_com_info['intercommunity_weight'], ref_com_weight_diff, input_com_weight_diff

    def SummaryPlot(self):

        # Data - strength
        in_reference_strengths = cu.node_strength_list(self.reference_graph, True)
        out_reference_strengths = cu.node_strength_list(self.reference_graph, False)
        in_input_strengths = cu.node_strength_list(self.input_graph, True)
        out_input_strengths = cu.node_strength_list(self.input_graph, False)

        sin_ref_x, sin_ref_y = cu.eccdf(in_reference_strengths)
        sin_in_x, sin_in_y = cu.eccdf(in_input_strengths)

        sout_ref_x, sout_ref_y = cu.eccdf(out_reference_strengths)
        sout_in_x, sout_in_y = cu.eccdf(out_input_strengths)

        # Data - degree
        in_reference_degrees = self.reference_graph.in_degree(self.reference_graph.nodes(), weight=None).values()
        in_input_degrees = self.input_graph.in_degree(self.input_graph.nodes(), weight=None).values()
        out_reference_degrees = self.reference_graph.out_degree(self.reference_graph.nodes(), weight=None).values()
        out_input_degrees = self.input_graph.out_degree(self.input_graph.nodes(), weight=None).values()

        din_ref_x, din_ref_y = cu.eccdf(in_reference_degrees)
        din_in_x, din_in_y = cu.eccdf(in_input_degrees)

        dout_ref_x, dout_ref_y = cu.eccdf(out_reference_degrees)
        dout_in_x, dout_in_y = cu.eccdf(out_input_degrees)

        # Data - sdiff
        reference_diff = np.abs(np.array(in_reference_strengths) - np.array(out_reference_strengths))
        input_diff = np.abs(np.array(in_input_strengths) - np.array(out_input_strengths))

        sdiff_ref_x, sdiff_ref_y = cu.eccdf(reference_diff)
        sdiff_in_x, sdiff_in_y = cu.eccdf(input_diff)

        total_weight_diff, intracommunity_weight_diff, target_mu, mu, ref_com_weight_diff, input_com_weight_diff = self.CheckCommunityDifferences()

        pltText_com_info = ""
        for com in ref_com_weight_diff:
            pltText_com_info += str(com) + " r: " + str(round(ref_com_weight_diff[com],2)) + " m: "+ str(round(input_com_weight_diff[com],2)) + "\n"

        # Plot
        f, axarr = plt.subplots(3)
        f.set_size_inches(6, 12)
        axarr[0].set_title(" mu: " + str(mu))

        axarr[0].plot(sin_ref_x, sin_ref_y, color='black', marker='o', label='Target in-str', mfc='none', markeredgecolor='black', ls='none')
        axarr[0].plot(sin_in_x, sin_in_y, color='red', marker='o', label='Result in-str', mfc='none', markeredgecolor='red', ls='none')
        axarr[0].plot(sout_ref_x, sout_ref_y, color='black', marker='x', label='Target out-str', ls='none')
        axarr[0].plot(sout_in_x, sout_in_y, color='red', marker='x', label='Result out-str', ls='none')
        axarr[0].set_ylabel('CCDF')
        axarr[0].set_xlabel('log(strength)')
        axarr[0].legend(loc="lower left")

        axarr[1].plot(din_ref_x, din_ref_y, color='black', marker='o', label='Target in-deg', mfc='none', markeredgecolor='black', ls='none')
        axarr[1].plot(din_in_x, din_in_y, color='red', marker='o', label='Result in-deg', mfc='none', markeredgecolor='red', ls='none')
        axarr[1].plot(dout_ref_x, dout_ref_y, color='black', marker='x', label='Target out-deg', ls='none')
        axarr[1].plot(dout_in_x, dout_in_y, color='red', marker='x', label='Result out-deg', ls='none')
        axarr[1].set_ylabel('CCDF')
        axarr[1].set_xlabel('degree')
        axarr[1].legend()

        axarr[2].annotate(pltText_com_info, xy=(0.7, 0.3), fontsize=9, xycoords='axes fraction')
        axarr[2].plot(sdiff_ref_x, sdiff_ref_y, color='black', marker='o', label='Target sdiff', mfc='none', markeredgecolor='black', ls='none')
        axarr[2].plot(sdiff_in_x, sdiff_in_y, color='red', marker='o', label='Result sdiff', mfc='none', markeredgecolor='red', ls='none')
        axarr[2].set_ylabel('CCDF')
        axarr[2].set_xlabel('in/out difference')
        axarr[2].legend()

        plt.tight_layout()
        plt.savefig(self.prefix + "_summary.png", dpi=300)
        plt.close()
        plt.clf()

if __name__ == '__main__':
    """
    Testing
    """
    pass