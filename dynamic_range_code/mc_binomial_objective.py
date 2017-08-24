import networkx as nx
import graphgen
import utilities
import echostatenetwork
import numpy as np
from collections import Sequence

class mc_binomial_objective(object):

    def __init__(self, **kwargs):
        """
        
        Task parameters:
        p - probability of success (1)
        sequence_length - length of task
        cut - remove transient
        max_delay - how many delay's to go out to when calculating memory
        shift

        ESN parameters:

        input_fraction
        input_gain
        neuron_type
        neuron_pars
        output_neuron_type
        output_neuron_pars

        Network parameters: 

        N
        mu
        k
        maxk
        minc
        maxc
        deg_exp
        temp_dir_ID
        full_path
        reservoir_weight_scale
        spectral_factor
        input_weight_bounds
        lower_reservoir_bound
        upper_reservoir_bound

        """

        parameter_defaults = {
            'p':0.5,
            'training_length': 10,
            'validation_length': 10,
            'cut': 0,
            'max_delay': 100,
            'shift': 1,
            'input_fraction': 0.2,
            'input_gain': 1.0,
            'neuron_type': 'sigmoid',
            'neuron_pars': {}, 
            'output_neuron_type': 'identity',
            'output_neuron_pars': {},
            'N': 1000, 
            'mu': 0.5,
            'k': 6, 
            'maxk': 6,
            'minc':10, 
            'maxc':10, 
            'deg_exp':1.0, 
            'temp_dir_ID':0, 
            'full_path':None,
            'reservoir_weight_scale': 1.0,
            'spectral_factor': None, # Overrides reservoir_weigh_scale
            'input_weight_bounds': (0.0, 1.0),
            'lower_reservoir_bound': 0.0,
            'upper_reservoir_bound': 1.0}

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.esn = self.generate_esn()

    def calculate_largest_eigenval(self, adj_matrix):
        
        from numpy import linalg

        try:
            eigenvalues = linalg.eigvals(adj_matrix)
        except linalg.LinAlgError:
            print "Error: Failure to converge matrix"
            sys.exit()

        largest_eigenvalue = np.real(eigenvalues).max()

        return largest_eigenvalue

    def generator_reservoir(self):
        """
        Use another function to create the reservoir and return it
        """

        self.nx_reservoir = graphgen.uniform_weighted_directed_lfr_graph(N=self.N, mu=self.mu, 
            k=self.k, maxk=self.maxk, minc=self.minc, maxc=self.maxc, deg_exp=self.deg_exp, 
            temp_dir_ID=self.temp_dir_ID, full_path=self.full_path, 
            weight_bounds=self.reservoir_weight_scale * np.array([self.lower_reservoir_bound, self.upper_reservoir_bound]))

        reservoir = np.asarray(nx.to_numpy_matrix(self.nx_reservoir))

        if self.spectral_factor != None:
            reservoir =  np.asmatrix(reservoir) * np.identity(self.N)
            effective_spectral_radius = self.calculate_largest_eigenval(reservoir)
            print effective_spectral_radius
            reservoir = reservoir / effective_spectral_radius * self.spectral_factor

        return reservoir

    def generate_input_weights(self):
        """
        Generates the input weights for the reservoir. It does this by generating first an array
        of either random or unitary array and then applying a mask to reduce the inputs down to
        that required by the input fraction. If a sequence of input gains are given, then the
        corresponding dimension's weights are adjusted by that gain, else if a scalar is given
        the whole array is adjusted.
        """

        input_weights = np.random.uniform(self.input_weight_bounds[0], self.input_weight_bounds[1], \
            size=(self.N, 1))
        
        if isinstance(self.input_gain, Sequence):
            for i in input_weights.shape[1]:
                input_weights[:,i] *= self.input_gain[i]
        else:
            input_weights *= self.input_gain

        fraction_mask = np.zeros(self.N * 1)
        fraction_mask[:int(self.input_fraction * self.N)] = 1.0
        np.random.shuffle(fraction_mask)
        fraction_mask = fraction_mask.reshape((self.N, 1))
        input_weights = input_weights * fraction_mask

        return input_weights

    def generate_esn(self):
        """
        """

        self.reservoir = self.generator_reservoir()

        self.input_weights = self.generate_input_weights()
        return echostatenetwork.DESN(self.reservoir, self.input_weights, 
            neuron_type=self.neuron_type, output_type=self.output_neuron_type, 
            neuron_pars=self.neuron_pars, output_neuron_pars=self.output_neuron_pars,
            init_state="zeros")

    def generate_signal(self, length):
        """
        input signal is: T x 1
        target is: T x int(max_delay / shift)
        """

        self.sequence_length = self.cut + length + self.max_delay
        input_signal = np.random.binomial(1 , self.p, size=(self.sequence_length, 1))

        target_signal = np.zeros((length, int(self.max_delay / self.shift)))
        for i in range(target_signal.shape[1]):
            target_signal[:,i] = input_signal[self.cut + (i+1)*self.shift : length + self.cut + (i+1)*self.shift, 0]

        return input_signal, target_signal

    def generate_input(self, length):
        """
        Adjust shape for ESN by putting it in columnar format
        """

        input_signal, target_signal = self.generate_signal(length)
        # reshape input into be compatible with ESN
        input_signal = input_signal.reshape((input_signal.shape[0], input_signal.shape[1], 1))
        target_signal = target_signal.reshape((target_signal.shape[0], target_signal.shape[1], 1))

        return input_signal, target_signal

    def train(self):
        """
        """

        self.esn.Train(*self.generate_input(self.training_length), cut=self.cut+self.max_delay, cut_target_output=False)
        self.esn.Reset()

    def validate(self):
        """
        """

        # Run validation test
        test_input, test_targets = self.generate_input(self.validation_length)
        self.esn.Reset()
        _, prediction, test_targets, _ = self.esn.Predict(test_input, test_targets, cut=self.cut+self.max_delay, 
            cut_target_output=False, error_type=None, analysis_mode=True)
        self.network_history_time_by_node = np.array(self.esn.network_history)[:,:,0].copy()
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

    def calculate_avg_activity(self):
        """
        requires validation to have been run
        """

        history_avg_over_nodes = np.mean(self.network_history_time_by_node, axis=1)
        return np.mean(history_avg_over_nodes[self.cut:])

    def get_community_activity(self):

        # Determine community activities
        dictCom_listNodes = utilities.get_community_node_dict(self.nx_reservoir, 'community')
        dictCom_listActivity = {}
        for com, nodes in dictCom_listNodes.items():
            # 1 is subtracted because the ids start from 1, while the indexes from 0
            community_state = self.network_history_time_by_node[:,np.array(nodes)-1]
            dictCom_listActivity[com] = np.mean(community_state, axis=1)

        return dictCom_listActivity

    def plot_periodogram(self, prefix):
        """
        agregates network activity and uses that time-series to
        make a periodogram
        """
        import matplotlib.pyplot as plt
        from scipy import signal

        arTime_avgState = np.mean(self.network_history_time_by_node, axis=1)
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
        import matplotlib.pyplot as plt

        arTime_avgState = np.mean(self.network_history_time_by_node, axis=1)
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
        import matplotlib.pyplot as plt

        arTime_avgState = np.mean(self.network_history_time_by_node, axis=1)[self.cut:]

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
        import matplotlib.pyplot as plt

        f, ax = plt.subplots(7, 1, figsize=(7,7), sharey=True, sharex=True)
        arrNode_arrTime = self.network_history_time_by_node.T
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
        import matplotlib.pyplot as plt

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

    import matplotlib.pyplot as plt
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
    mc_obj = mc_binomial_objective(training_length=6000, validation_length=200,
        cut = 500, max_delay = 25, shift=1, reservoir_weight_scale=1.132,
        input_fraction=0.5, input_gain=1.0, output_neuron_type='heaviside',
        output_neuron_pars={'threshold': 0.5},
        neuron_type='sigmoid', neuron_pars={'c':1, 'e':10}, p=0.5,
        mu=0.25, N=500, input_weight_bounds=(-0.2,1.0), lower_reservoir_bound=-0.2, 
        upper_reservoir_bound=1.0, k=6, maxk=6, minc=10, maxc=10, spectral_factor=None)
    mc_obj.train()
    print mc_obj.validate()
    print mc_obj.calculate_avg_activity()
    mc_obj.plot_community_activity_timeseries('test')
    mc_obj.plot_neuron_activity_timeseries('test')
    mc_obj.plot_periodogram('test')
    mc_obj.plot_total_activity('test')
    mc_obj.plot_lag_activity('test', [1,2], 3)