import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import echostatenetwork
import copy
import sys

class DiscreteEchoStateModel(object):
    """
    Echo state network modelled on a two community network
    """

    def __init__(self, network, input_weight_scale, signal_ratio, 
        niter, input_series, neuron_type="tanh", initialization_state="zeros", 
        neuron_pars=None, input_weight_bounds=(-1.0, 1.0), community_key=None, 
        seed=1, target_community=None):
        """
        input_series needs to be a Lx1x1 vector or a list of Kx1 vectors. L is the length of the input. Simulation time exceeding L will assume inputs of 0
        """

        self.network = network
        self.signal_ratio = signal_ratio
        self.input_weight_scale = input_weight_scale
        self.state_history = []
        self.initialization_state = initialization_state
        self.niter = niter
        self.default_input = np.zeros((1, 1))
        self.neuron_type = neuron_type
        self.neuron_pars = neuron_pars
        self.input_series = input_series
        self.input_weight_bounds = input_weight_bounds
        self.community_key = community_key
        self.target_community = target_community
        
        # Generate input weights
        self.input_weights = self.CreateInputWeights(target_community=self.target_community, community_key=self.community_key)
        self.reservoir = nx.adjacency_matrix(self.network).A # toarray() may need to be added for different versions of networkx
        self.esn_model = echostatenetwork.DESN(self.reservoir, self.input_weights, self.neuron_type, init_state=self.initialization_state, neuron_pars=self.neuron_pars)

    def Run(self):
        """
        """
        self.state_history.append(copy.copy(self.esn_model.current_state))
        for i in xrange(self.niter):
            if i < len(self.input_series):
                self.esn_model.Step(self.input_series[i], record=True)
            else:
                self.esn_model.Step(self.default_input, record=True)

        # print len(self.state_history), len(esn_model.network_history)
        self.state_history += self.esn_model.network_history

    def net_activity(self):
        """
        """

        return np.sum(np.abs(self.state_history))

    def last_activity(self):
        """
        Returns last value taken by model
        """

        return np.sum(np.abs(self.state_history[-1]))

    def community_activity(self, community, community_key):
        """
        """

        network_nodes = self.network.nodes()
        community_nodes = set(self.GetCommunityNodes(community_key, community))
        community_mask = np.array([ True if node in community_nodes else False for node in network_nodes ])

        return np.sum([ np.abs(state[community_mask]) for state in self.state_history ])

    def community_last_activity(self, community, community_key):
        """
        """

        network_nodes = self.network.nodes()
        community_nodes = set(self.GetCommunityNodes(community_key, community))
        community_mask = np.array([ True if node in community_nodes else False for node in network_nodes ])

        return np.sum(np.abs(self.state_history[-1][community_mask]))

    def net_activation_time(self, threshold):
        """
        """

        for t, state in enumerate(self.state_history):
            if (np.sum(np.abs(state)) / float(len(self.network))) > threshold:
                return t

        return -1

    def community_activation_time(self, community, community_key, threshold):
        """
        """

        network_nodes = self.network.nodes()
        community_nodes = set(self.GetCommunityNodes(community_key, community))
        community_mask = np.array([ True if node in community_nodes else False for node in network_nodes ])
        for t, state in enumerate(self.state_history):
            if (np.sum(np.abs(state[community_mask])) / float(len(community_nodes))) > threshold:
                return t 

        return -1

    def net_halting_time(self, low_bound = 0.001):
        """
        """

        for t, state in reversed(list(enumerate(self.state_history))):
            if (np.sum(np.abs(state)) / float(len(self.network))) > low_bound:
                return t

        return 0

    def community_halting_time(self, community, community_key, low_bound = 0.001):
        """
        """

        network_nodes = self.network.nodes()
        community_nodes = set(self.GetCommunityNodes(community_key, community))
        community_mask = np.array([ True if node in community_nodes else False for node in network_nodes ])
        for t, state in reversed(list(enumerate(self.state_history))):
            if (np.sum(np.abs(state[community_mask])) / float(len(community_nodes))) > low_bound:
                return t

        return 0

    def CreateInputWeights(self, target_community=None, community_key=None):
        """
        """

        num_inputs = int(len(self.network) * self.signal_ratio)
        if num_inputs < 1:
            num_inputs = 1

        if (target_community != None) and (community_key != None):
            if hasattr(target_community, '__iter__'):
                input_neurons = []
                for com in target_community:
                    input_neurons += self.GetCommunityNodes(community_key, com)
            else:
                input_neurons = self.GetCommunityNodes(community_key, target_community)
        else:
            input_neurons = self.network.nodes()

        if len(input_neurons) < num_inputs:
            print "Warning: Only ", str(input_neurons), " available. Require ", str(num_inputs), ". Inputs set to max available."
            num_inputs = len(input_neurons)
        input_neurons = np.random.choice(input_neurons, size=num_inputs, replace=False)

        dictNode_index = self.GenerateNodeIDtoReservoirIndexDict(True)
        input_mask = np.zeros((len(self.network), 1))
        for node in input_neurons:
            input_mask[dictNode_index[node],0] = 1

        return self.input_weight_scale * np.random.uniform(self.input_weight_bounds[0], self.input_weight_bounds[1], size=(len(self.network), 1)) * input_mask

    def GenerateNodeIDtoReservoirIndexDict(self, node_to_index=True):
        """
        """

        if node_to_index:
            return { node : i for i, node in enumerate(self.network.nodes()) }
        else:
            return { i : node for i, node in enumerate(self.network.nodes()) }

    def GetCommunityNodes(self, community_key, target_community):
        """
        """

        community_nodes = []
        for node in self.network.nodes():
            community = self.network.node[node][community_key]
            if community == target_community:
                community_nodes.append(node)

        return community_nodes

    def GetCommunityNodeDict(self, community_key):
        """
        Returns the communities found in the graph using a dictionary with keys
        for whatever the community was named.
        """

        community_dict = {}
        for node in self.network.nodes():
            community = self.network.node[node][community_key]
            if community in community_dict:
                community_dict[community].append(node)
            else:
                community_dict[community] = [node]

        return community_dict

    def time_series_activity_plot(self, prefix):
        """
        """

        if (self.community_key != None) and (self.target_community != None):
            for community in self.GetCommunityNodeDict(self.community_key).keys():
                network_nodes = self.network.nodes()
                community_nodes = set(self.GetCommunityNodes(self.community_key, community))
                community_mask = np.array([ True if node in community_nodes else False for node in network_nodes ])
                list_activities = []
                for t, state in enumerate(self.state_history):
                    list_activities.append(np.sum(np.abs(state[community_mask])) / float(len(community_nodes)))
                times = xrange(len(self.state_history))

                plt.plot(times, list_activities, marker='o', ls='-', label='Community ' + str(community))

            list_activities = []
            for t, state in enumerate(self.state_history):
                list_activities.append(np.sum(np.abs(state)) / float(len(self.network)))
            times = xrange(len(self.state_history))

            plt.plot(times, list_activities, color='black', marker='o', ls='-', lw=2, label='Network')
            plt.xlabel("time")
            plt.ylabel("Fractional activation")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(prefix + "_time-series_activity_plot.png", dpi=300)
            plt.clf()
            plt.close()

if __name__ == '__main__':
    """
    testing
    """
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

    import graphgen
    def adjust_graph_spectral_radius(graph, new_spectral_radius):

        largest_eigenval = max(np.abs(np.real(np.linalg.eigvals(nx.adjacency_matrix(graph).A))))
        for edge in graph.edges_iter():
            graph[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight'] / largest_eigenval * new_spectral_radius

    N = 500
    mu = 0.005
    avgD = 7
    graph = graphgen.uniform_weighted_two_community_graph(N, mu, avgD, 0.56666667, 0.56666667)
    # adjust_graph_spectral_radius(graph, 0.90)
    signal_weight = 40.0
    signal_ratio = 0.2#.01 .2
    tmax = 100
    testDESM = DiscreteEchoStateModel(graph, signal_weight, signal_ratio, tmax, np.ones((1,1,1)), neuron_type='sigmoid', 
        neuron_pars={'e':10, 'c':1}, initialization_state=(0.0,0.0), input_weight_bounds=(1.0,1.0),
        seed=1,  community_key='community', target_community=1) #np.append(np.zeros((20,1,1)), np.ones((5,1,1)), axis=0)
    testDESM.Run()
    testDESM.time_series_activity_plot('test')