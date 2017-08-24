import networkx as nx
import graphgen 
import utilities
import matplotlib.pyplot as plt

def make_example_graphs():

    # results = utilities.load_object('/home/nathaniel/workspace/DiscreteESN/nbit_task/3bit_T50_N500_c10_in0-1_thresh_Ws1.5/thresh_T50_N500_c10_in0-1_Ws1.5_mu-rsig_final_results.pyobj')
    # print results.keys()
    # print results['parameters']

    mu = 0.1
    best_graph = graphgen.uniform_weighted_directed_lfr_graph_asnx(N=500, mu=mu, 
        k=6, maxk=6, maxc=10, minc=10, weight_bounds=(1.0,1.0))
    nx.write_gexf(best_graph, '50com_reservoir.gexf')

if __name__ == '__main__':
    """
    """
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

    make_example_graphs()