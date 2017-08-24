import networkx as nx
import numpy as np
import graphgen 
import utilities
import matplotlib.pyplot as plt
import memory_capacity_objective
from sklearn.decomposition import PCA
import sys

class mc_analyzer(memory_capacity_objective.memory_capacity_objective):
    """
    """

    def __init__(self, **kwargs):
        super(mc_analyzer, self).__init__(**kwargs)

    def validate(self):
        """
        """

        # Run validation test
        test_input, test_targets = self.generate_input()
        self.esn.Reset()
        _, prediction, test_targets, _ = self.esn.Predict(test_input, test_targets, cut=self.cut+self.max_delay, 
            cut_target_output=False, error_type=None, analysis_mode=True)
        self.arrTime_state = np.array(self.esn.network_history)[:,:,0].copy()
        self.esn.Reset()

        # Evaluate correlation coefficient for all delays
        num_delays = int(self.max_delay / self.shift)
        arr_delay = np.array([ i * self.shift for i in reversed(range(1, num_delays + 1)) ])
        arr_detcoef = np.zeros(num_delays)
        for i in range(num_delays):
            cor_coef = np.corrcoef(prediction[:,i], test_targets[:,i])[0,1]
            if np.isnan(cor_coef):
                arr_detcoef[i] = 0.0
            else:
                arr_detcoef[i] = cor_coef**2

        MC = np.sum(arr_detcoef) * self.shift

        return MC, arr_delay, arr_detcoef

    def get_community_activity(self):

        # Determine community activities
        dictCom_listNodes = utilities.get_community_node_dict(self.nx_reservoir, 'community')
        dictCom_listActivity = {}
        for com, nodes in dictCom_listNodes.items():
            # 1 is subtracted because the ids start from 1, while the indexes from 0
            community_state = self.arrTime_state[:,np.array(nodes)-1]
            dictCom_listActivity[com] = np.mean(community_state, axis=1)

        return dictCom_listActivity

    def plot_periodogram(self, prefix):
        """
        agregates network activity and uses that time-series to
        make a periodogram
        """

        from scipy import signal

        arTime_avgState = np.mean(self.arrTime_state, axis=1)
        f, Pxx_den = signal.periodogram(arTime_avgState[self.cut:])
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD')
        plt.ylim(ymin=10**-8)
        plt.tight_layout()
        plt.savefig(prefix + "_power_spectrum.png", dpi=300)
        plt.close()
        plt.clf()

    def plot_total_activity(self, prefix):
        """
        take mean activity over whole network
        """

        arTime_avgState = np.mean(self.arrTime_state, axis=1)
        plt.plot(arTime_avgState)
        plt.xlabel("time")
        plt.ylabel("activity")
        plt.xlim(xmax=len(arTime_avgState))
        plt.tight_layout()
        plt.savefig(prefix + "_net_activity_ts.png", dpi=300)
        plt.clf()
        plt.close()

    def plot_lag_activity(self, prefix, tau=1, dimension=2):
        """
        plot F[t] vs F[t-tau]
        """

        arTime_avgState = np.mean(self.arrTime_state, axis=1)[self.cut:]

        if dimension == 2:
            F_t = arTime_avgState[:-tau]
            F_tau = arTime_avgState[tau:]
            plt.scatter(F_t, F_tau)

        elif dimension == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            F_x = arTime_avgState[:-tau[1]]
            F_y = arTime_avgState[tau[0]:-(tau[1]-tau[0])]
            F_z = arTime_avgState[tau[1]:]
            ax.scatter(F_x, F_y, F_z)
            ax.set_xlabel('A[t]')
            ax.set_ylabel('A[t-'+str(tau[0])+']')
            ax.set_zlabel('A[t-'+str(tau[1])+']')

        plt.tight_layout()
        plt.savefig(prefix + "_lag_activity_ts.png", dpi=300)
        plt.clf()
        plt.close()

    def plot_neuron_activity_timeseries(self, prefix):
        """
        """

        f, ax = plt.subplots(7, 1, figsize=(7,7), sharey=True, sharex=True)
        arrNode_arrTime = self.arrTime_state.T
        for node in range(7):#self.N:
            ax[node].plot(arrNode_arrTime[node])
            ax[node].tick_params(axis='y', labelsize=10)

        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlim(xmax=len(arrNode_arrTime[0]))
        plt.savefig(prefix + "_neuron_traj.png", dpi=300)
        plt.close()
        plt.clf()

    def plot_community_activity_timeseries(self, prefix):
        """
        """

        dictCom_listActivity = self.get_community_activity()

        # Plot activity over time
        for com, activity in dictCom_listActivity.items():
            # plt.clf()
            plt.plot(range(1,len(activity)+1), activity, lw=1, ls='-')

        plt.xlabel("time")
        plt.ylabel("activity")
        plt.xlim(xmax=len(activity+1))
        plt.legend(loc=2)
        plt.tight_layout()
        plt.savefig(prefix + "_com_activity_ts.png", dpi=300)
        plt.clf()
        plt.close()

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

    np.set_printoptions(suppress=True)
    mc_obj = mc_analyzer(training_length=1500, validation_length=1500,
        cut = 500, max_delay = 25, shift=1, binary=True, reservoir_weight_scale=1,#1.132,
        input_fraction=0.3, input_gain=1.0, output_neuron_type='heaviside',
        output_neuron_pars={'threshold': 0.5}, input_range= (0.0, 1.0),
        neuron_type='sigmoid', neuron_pars={'c':1, 'e':10},
        mu=0.25, N=500, input_weight_bounds=(-0.2,1.0), lower_reservoir_bound=-0.2, 
        upper_reservoir_bound=1.0, k=6, maxk=6, minc=10, maxc=10)
    mc_obj.train()
    print mc_obj.validate()
    mc_obj.plot_community_activity_timeseries('test')
    mc_obj.plot_neuron_activity_timeseries('test')
    mc_obj.plot_periodogram('test')
    mc_obj.plot_total_activity('test')
    mc_obj.plot_lag_activity('test', [1,2], 3)