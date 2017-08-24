import inspect
import multicommunityLFR
import networkx as nx
import numpy as np

def make_reservoir(N, mu, k, maxk, minc, maxc, weight_bounds, **kwargs):
    """
    """

    multicommunityLFR_pardict = { key: value for key, value in kwargs.items() \
        if key in inspect.getargspec(multicommunityLFR.make_lfr_graph).args }

    reservoir = multicommunityLFR.make_lfr_graph(N, mu, k, maxk, minc, maxc, **multicommunityLFR_pardict)
    weights = np.random.uniform(weight_bounds[0], weight_bounds[1], size=nx.number_of_edges(reservoir))
    for i, edge in enumerate(reservoir.edges_iter()):
        reservoir[edge[0]][edge[1]]['weight'] = weights[i]

    return reservoir

def make_reservoir_asarray(**kwargs):
    """
    """

    return np.asarray(nx.to_numpy_matrix(make_reservoir(**kwargs)))


def make_reservoir_asnx(**kwargs):
    """
    """

    return make_reservoir(**kwargs)

if __name__ == '__main__':
    
    print make_reservoir_asarray(N=50, mu=0.1, k=4, maxk=4, minc=25, maxc=25, weight_bounds=(0.5, 1.0))