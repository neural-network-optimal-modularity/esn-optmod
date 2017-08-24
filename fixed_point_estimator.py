import echostatenetwork
import numpy as np

def fixed_point_estimate(esn, num_attempts, run_time, distance_thresh=0.1, 
    epsilon=0.001, convergence_delay=10):
    """
    determined the fraction of divergent trajectories, where divergence means that the last time
    step has a larger difference with the last 'convergence_delay' times steps than epsilon.

    if also returns the number of unique stable states (which didn't diverge), the number of divergent states
    where dist_thresh is used to determine whether they are different.
    """
    if convergence_delay > run_time:
        print "Warning! Convergence_delay is greater than run_time: setting convergence_delay to run_time"
        convergence_delay = run_time

    stable_state_list = []
    num_diverged = 0.0
    for attempt in range(num_attempts):
        esn.Reset()
        esn.run_reservoir(run_time, record=True)
        if np.any(np.abs(esn.network_history[-1] - np.array(esn.network_history[-convergence_delay:-2])) > epsilon):
            num_diverged += 1
        else:
            # If it didn't diverge, then add it to the stable list
            stable_state_list.append(np.reshape(esn.network_history[-1], (esn.num_neurons)))

    unique_state_list = []
    while stable_state_list:
        for state in unique_state_list:
            if np.linalg.norm(stable_state_list[-1] - state) < distance_thresh:
                stable_state_list.pop()
                break
        
        else:
            unique_state_list.append(stable_state_list.pop())

    return len(unique_state_list), num_diverged

if __name__ == '__main__':
    """test"""

    # import graphgen
    # import networkx as nx

    # np.random.seed()

    # reservoir = np.asarray(nx.to_numpy_matrix(graphgen.uniform_weighted_directed_lfr_graph(N=500, 
    #     mu=0.3, k=6, maxk=6, minc=10, maxc=10, deg_exp=1.0, 
    #     temp_dir_ID=str(np.random.randint(0,1000))+'temptest', full_path="/home/nathaniel/workspace/DiscreteESN/", #/N/u/njrodrig/BigRed2/topology_of_function/
    #     weight_bounds=(-0.2,1.0))))

    # esn = echostatenetwork.DESN(reservoir, neuron_type='sigmoid', 
    #     neuron_pars={'c':1, 'e':10}, init_state=(0.2,1.0))

    # print fixed_point_estimate(esn, 1000, 1000, distance_thresh=0.01, epsilon=0.01, convergence_delay=10)

    # Run plots
    import matplotlib.pyplot as plt
    import utilities
    import graphgen
    import networkx as nx

    def get_community_activity(nx_reservoir, states):

        # Determine community activities
        dictCom_listNodes = utilities.get_community_node_dict(nx_reservoir, 'community')
        dictCom_listActivity = {}
        for com, nodes in dictCom_listNodes.items():
            # 1 is subtracted because the ids start from 1, while the indexes from 0
            community_state = states[:,np.array(nodes)-1]
            dictCom_listActivity[com] = np.mean(community_state, axis=1)

        return dictCom_listActivity

    def plot_community_activity_timeseries(prefix, nx_reservoir, states):
        """
        """

        dictCom_listActivity = get_community_activity(nx_reservoir, states)

        # Plot activity over time
        for com, activity in dictCom_listActivity.items():
            # plt.clf()
            plt.plot(range(1,len(activity)+1), activity, lw=1, ls='-')

        plt.xlabel("time")
        plt.ylabel("activity")
        plt.legend(loc=2)
        plt.xlim(1,len(activity)+1)
        plt.tight_layout()
        plt.savefig(prefix + "_com_activity_ts.png", dpi=300)
        plt.clf()
        plt.close()

    nx_reservoir = graphgen.uniform_weighted_directed_lfr_graph(N=500, 
        mu=0.01, k=6, maxk=6, minc=10, maxc=10, deg_exp=1.0, 
        temp_dir_ID=str(np.random.randint(0,1000))+'temptest', full_path="/home/nathaniel/workspace/DiscreteESN/", #/N/u/njrodrig/BigRed2/topology_of_function/
        weight_bounds=(-0.2,1.0))
    reservoir = np.asarray(nx.to_numpy_matrix(nx_reservoir))
    esn = echostatenetwork.DESN(reservoir, neuron_type='sigmoid', 
        neuron_pars={'c':1, 'e':10}, init_state=(0.0,0.6))
    esn.run_reservoir(100, record=True)
    state_history = np.array(esn.network_history)
    state_history = state_history.reshape((state_history.shape[0], state_history.shape[1]))
    plot_community_activity_timeseries("test", nx_reservoir, state_history)