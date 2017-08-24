import numpy as np 
import networkx as nx 
import graphgen

if __name__ == '__main__':
    """
    """

    graph = graphgen.uniform_weighted_directed_lfr_graph(N=40, mu=0.5, 
            k=5, maxk=5, minc=10, maxc=10, deg_exp=1, 
            temp_dir_ID=0, full_path=None, 
            weight_bounds=1 * np.array([1, 1]))
    nx.write_gexf(graph, "high_mu_graph.gexf")