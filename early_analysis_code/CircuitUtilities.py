import sys
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

def save_object(obj, filename):
    """
    Save an object to file for later use.
    """
    
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()

def load_object(filename):
    """
    Load a previously saved object.
    """
    
    file = open(filename, 'rb')
    return pickle.load(file)

def get_communities(graph, community_key):
    # Get community list
    communities = []
    for node in graph.nodes():
        communities.append(graph.node[node][community_key])

    # Remove copies
    communities = set(communities)

    return communities

def get_community_node_dict(graph, community_key=None, communities=None):
    # Get community list
    if communities == None and community_key != None:
        communities = get_communities(graph, community_key)

    # Generate dict
    community_nodes = {}
    for community in communities:
        for node in graph.nodes_iter():
            if graph.node[node][community_key] == community:
                try:
                    community_nodes[community].append(node)
                except KeyError:
                    community_nodes[community] = [node]

    return community_nodes

def get_bridge_dict(graph, community_key):
    # Generate bridge dict
    bridge_dict = {}
    for edge in graph.edges_iter():
        if graph.node[edge[0]][community_key] != graph.node[edge[1]][community_key]:
            try:
                bridge_dict[(graph.node[edge[0]][community_key], graph.node[edge[1]][community_key])].append(edge[0])
            except KeyError:
                bridge_dict[(graph.node[edge[0]][community_key], graph.node[edge[1]][community_key])] = [edge[0]]

    return bridge_dict

def calculate_muw(graph, community_key):

    total_weight = 0.0
    bridge_weight = 0.0
    for edge in graph.edges_iter():
        if graph.node[edge[0]][community_key] != graph.node[edge[1]][community_key]:
            bridge_weight += graph[edge[0]][edge[1]]['weight']
            total_weight += graph[edge[0]][edge[1]]['weight']
        else:
            total_weight += graph[edge[0]][edge[1]]['weight']

    return bridge_weight / total_weight

def bridge_edge_set(graph, community_key):

    bridge_edge_list = []
    for edge in graph.edges_iter():
        if graph.node[edge[0]][community_key] != graph.node[edge[1]][community_key]:
            bridge_edge_list.append(edge)

    return set(bridge_edge_list)

def community_mu(graph, community_key):
    """
    """

    # Get community list
    communities = []
    for node in graph.nodes():
        communities.append(graph.node[node][community_key])

    # Remove copies
    communities = set(communities)

    # Community nodes lists
    community_node_lists = {}
    community_sizes = {}
    for community in communities:
        community_node_lists[community] = get_nodes_list(graph, community, community_key)
        community_sizes[community] = len(community_node_lists[community]) / float(len(graph))

    # Get ride on crap communities
    cleanse_heretic_communities(communities, community_node_lists, graph, community_key)

    # Get community in_mu and out_mu
    community_mu_dict = {}
    for community in communities:
        out_mu, in_mu = ratio_bridges_community(graph, community_node_lists[community], community_key)
        community_mu_dict[community] = {'out':out_mu, 'in':in_mu}

    return community_mu_dict, community_sizes

def graph_mu(graph, community_key):

    # Get communities
    communities = get_communities(graph, community_key)
    # Get nodes in each community
    dictCommunities_listNodes = get_community_node_dict(graph, community_key) 

    intracommunity_weight = 0.0
    intercommunity_weight = 0.0
    total_weight = 0.0
    for edge in graph.edges_iter():

        # check if edge is within or between communities
        if graph.node[edge[0]][community_key] == graph.node[edge[1]][community_key]:
            intracommunity_weight += graph[edge[0]][edge[1]]['weight']
        else:
            intercommunity_weight += graph[edge[0]][edge[1]]['weight']

        total_weight += graph[edge[0]][edge[1]]['weight']

    return intercommunity_weight / total_weight

def get_nodes_list(graph, community, community_key):
    """
    """

    return [ node for node in graph.nodes() if graph.node[node][community_key]==community ]

def community_information(graph, community_key):
    """
    """

    # Get community list
    communities = []
    for node in graph.nodes():
        communities.append(graph.node[node][community_key])

    # Remove copies
    communities = set(communities)

    # Community nodes lists
    community_node_lists = {}
    for community in communities:
        community_node_lists[community] = get_nodes_list(graph, community, community_key)

    # get sizes for all communities in graph
    community_sizes = {}
    for community in community_node_lists.keys():
        community_sizes[community] = len(community_node_lists[community])

    return community_sizes, community_node_lists

def graph_edge_information(graph, community_key):
    """
    Assumes graph is weighted and directed
    Returns # edges between and within communities (and each ones)
    Returns Sum weights between and within communities (and each ones)
    Returns pairwise weight sums and # edges between communities
    """

    # Get communities
    communities = get_communities(graph, community_key)
    # Get nodes in each community
    dictCommunities_listNodes = get_community_node_dict(graph, community_key) # could be faster using set instead of node list

    # Loop through each edge and count it toward each of the quantities we want to know
    intercommunity_edges = 0
    intracommunity_edges = 0
    total_edges = 0
    intercommunity_weight = 0.0
    intracommunity_weight = 0.0
    total_weight = 0.0
    dictCommunities_edges = {}
    dictCommunities_weight = {}
    dictCommunities_entering = {}
    dictCommunities_leaving = {}
    dictCommunities_dictCommunities_weight = {} # Pairwise weight between communities
    for edge in graph.edges_iter():

        # check if edge is within or between communities
        if graph.node[edge[0]][community_key] == graph.node[edge[1]][community_key]:
            intracommunity_edges += 1
            intracommunity_weight += graph[edge[0]][edge[1]]['weight']
            # Add to weight within community
            try:
                dictCommunities_weight[graph.node[edge[0]][community_key]] += graph[edge[0]][edge[1]]['weight']
            except KeyError:
                dictCommunities_weight[graph.node[edge[0]][community_key]] = graph[edge[0]][edge[1]]['weight']
            # Add edges
            try:
                dictCommunities_edges[graph.node[edge[0]][community_key]] += 1
            except KeyError:
                dictCommunities_edges[graph.node[edge[0]][community_key]] = 1

        else:
            intercommunity_edges += 1
            intercommunity_weight += graph[edge[0]][edge[1]]['weight']

            # Add entering
            try:
                dictCommunities_entering[graph.node[edge[1]][community_key]] += graph[edge[0]][edge[1]]['weight']
            except KeyError:
                dictCommunities_entering[graph.node[edge[1]][community_key]] = graph[edge[0]][edge[1]]['weight']
            # Add leaving
            try:
                dictCommunities_leaving[graph.node[edge[0]][community_key]] += graph[edge[0]][edge[1]]['weight']
            except KeyError:
                dictCommunities_leaving[graph.node[edge[0]][community_key]] = graph[edge[0]][edge[1]]['weight']
            # Add pairwise weight
            try:
                dictCommunities_dictCommunities_weight[graph.node[edge[0]][community_key]][graph.node[edge[1]][community_key]] += graph[edge[0]][edge[1]]['weight']
            except KeyError:
                dictCommunities_dictCommunities_weight[graph.node[edge[0]][community_key]] = {}
                dictCommunities_dictCommunities_weight[graph.node[edge[0]][community_key]][graph.node[edge[1]][community_key]] = graph[edge[0]][edge[1]]['weight']

        total_edges += 1
        total_weight += graph[edge[0]][edge[1]]['weight']

    # Community sizes
    community_sizes = { community: len(dictCommunities_listNodes[community]) for community in list(communities) }

    return { 'intercommunity_fraction': intercommunity_edges / float(total_edges),
            'intracommunity_fraction': intracommunity_edges / float(total_edges),
            'total_edges': total_edges,
            'total_weight': total_weight,
            'intercommunity_weight': intercommunity_weight / total_weight,
            'intracommunity_weight': intracommunity_weight / total_weight,
            'dictCommunities_edges': dictCommunities_edges,
            'dictCommunities_weight': dictCommunities_weight,
            'dictCommunities_entering': dictCommunities_entering,
            'dictCommunities_leaving': dictCommunities_leaving,
            'dictCommunities_dictCommunities_weight': dictCommunities_dictCommunities_weight,
            'communities': communities,
            'community_sizes': community_sizes }

def community_weight_fraction(graph, community_key):
    # Get communities
    communities = get_communities(graph, community_key)

    # Get nodes in each community
    dictCommunities_listNodes = get_community_node_dict(graph, community_key) # could be faster using set instead of node list
    dictCommunities_weight = {}
    dictCommunity_weight_fraction = {}

    total_internal_weight = 0.0
    for edge in graph.edges_iter():
        # check if edge is within or between communities
        if graph.node[edge[0]][community_key] == graph.node[edge[1]][community_key]:
            total_internal_weight += graph[edge[0]][edge[1]]['weight']
            # Add to weight within community
            try:
                dictCommunities_weight[graph.node[edge[0]][community_key]] += graph[edge[0]][edge[1]]['weight']
            except KeyError:
                dictCommunities_weight[graph.node[edge[0]][community_key]] = graph[edge[0]][edge[1]]['weight']

    for community in communities:
        if not (community in dictCommunities_weight):
            dictCommunities_weight[community] = 0.0

    for community in communities:
        possible_com_connections = len(dictCommunities_listNodes[community]) * (len(dictCommunities_listNodes[community]) - 1.0)
        possible_total_connections = len(graph) * (len(graph) - 1.0)
        dictCommunity_weight_fraction[community] = dictCommunities_weight[community] / total_internal_weight * possible_total_connections / possible_com_connections

    return dictCommunity_weight_fraction

# def community_avg_internal_weight(graph, community_key):

    # Get communities
    communities = get_communities(graph, community_key)
    # Get nodes in each community
    dictCommunities_listNodes = get_community_node_dict(graph, community_key) # could be faster using set instead of node list
    dictCommunities_weight = {}
    dictCommunity_avg_internal_weight = {}

    for edge in graph.edges_iter():
        # check if edge is within or between communities
        if graph.node[edge[0]][community_key] == graph.node[edge[1]][community_key]:
            # Add to weight within community
            try:
                dictCommunities_weight[graph.node[edge[0]][community_key]] += graph[edge[0]][edge[1]]['weight']
            except KeyError:
                dictCommunities_weight[graph.node[edge[0]][community_key]] = graph[edge[0]][edge[1]]['weight']

    for community in communities:
        dictCommunity_avg_internal_weight[community] = dictCommunities_weight[community] / float(len(dictCommunities_listNodes[community]))

    return dictCommunity_avg_internal_weight

def node_weight_dict(graph, in_weight=True):
    """
    """

    if in_weight:
        # Loop through all edges
        node_in_weight_dict = {}
        for edge in graph.edges_iter():
            # Add weight to postsynaptic neuron's in_weights
            if node_in_weight_dict.has_key(edge[1]):
                node_in_weight_dict[edge[1]].append(graph.edge[edge[0]][edge[1]]['weight'])
            else:
                node_in_weight_dict[edge[1]] = [graph.edge[edge[0]][edge[1]]['weight']]

        # Loop through nodes and add unconnected ones
        for node in graph.nodes_iter():
            if not node_in_weight_dict.has_key(node):
                node_in_weight_dict[node] = 0.0

        return node_in_weight_dict

    else:
        node_out_weight_dict = {}
        for edge in graph.edges_iter():
            # Add weight to presynaptic neuron's out_weights
            if node_out_weight_dict.has_key(edge[0]):
                node_out_weight_dict[edge[0]].append(graph.edge[edge[0]][edge[1]]['weight'])
            else:
                node_out_weight_dict[edge[0]] = [graph.edge[edge[0]][edge[1]]['weight']]

        # Loop through nodes and add unconnected ones
        for node in graph.nodes_iter():
            if not node_out_weight_dict.has_key(node):
                node_out_weight_dict[node] = 0.0

        return node_out_weight_dict

def node_strength_list(graph, in_weight=True):

    dictNode_listWeights = node_weight_dict(graph, in_weight=in_weight)

    return [ np.sum(np.log(dictNode_listWeights[node])) \
        if dictNode_listWeights[node] > sys.float_info.min else math.log(sys.float_info.min) \
        for node in dictNode_listWeights.keys() ]

def global_node_weight_dict(graph, in_weight=True):
    """
    """

    if in_weight:
        # Loop through all edges
        node_in_weight_dict = {}
        for edge in graph.edges_iter():
            # Add weight to postsynaptic neuron's in_weights
            if node_in_weight_dict.has_key(graph.node[edge[1]]['global_id']):
                node_in_weight_dict[graph.node[edge[1]]['global_id']].append(graph.edge[edge[0]][edge[1]]['weight'])
            else:
                node_in_weight_dict[graph.node[edge[1]]['global_id']] = [graph.edge[edge[0]][edge[1]]['weight']]

        return node_in_weight_dict

    else:
        node_out_weight_dict = {}
        for edge in graph.edges_iter():
            # Add weight to presynaptic neuron's out_weights
            if node_out_weight_dict.has_key(graph.node[edge[0]]['global_id']):
                node_out_weight_dict[graph.node[edge[0]]['global_id']].append(graph.edge[edge[0]][edge[1]]['weight'])
            else:
                node_out_weight_dict[graph.node[edge[0]]['global_id']] = [graph.edge[edge[0]][edge[1]]['weight']]

        return node_out_weight_dict

def ecdf(data):

    sorted_data = np.sort(data)
    size = float(len(sorted_data))
    cdf = np.array([ i / size for i in xrange(1,len(sorted_data)+1) ])
    return sorted_data, cdf

def eccdf(data):

    sorted_data, cdf = ecdf(data)
    return sorted_data, 1. - cdf

def plot_ccdf(prefix, data, xlabel='', x_log=False, y_log=False):

    x, y = eccdf(data)
    plt.clf()
    plt.plot(x, y, 'bo')
    if x_log == True: plt.xscale('log')
    if y_log == True: plt.yscale('log')
    plt.ylabel('CCDF')
    plt.xlabel(xlabel)
    plt.savefig(prefix + '.png', dpi=300)
    plt.clf()
    plt.close()