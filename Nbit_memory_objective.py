import networkx as nx
import graphgen
import echostatenetwork
import Nbit_memory_task
import numpy as np
from collections import Sequence

class Nbit_memory_objective(object):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        parameter_defaults = {'distractor_value': 1,
            'cue_value': 1,
            'sequence_dimension': 2,
            'start_time': 0,
            'sequence_length': 1,
            'distraction_duration': 10,
            'distraction_range': None,
            'input_fraction': 0.1,
            'input_gain': 1.0,
            'num_trials': 1,
            'neuron_type': 'sigmoid',
            'neuron_pars': {}, 
            'output_neuron_type': 'heaviside',
            'output_neuron_pars': {'threshold': 0.5},
            'N': 1000, 
            'mu': 0.1,
            'k': 7, 
            'maxk': 7,
            'homok': None,
            'com_size': None, 
            'minc':10, 
            'maxc':10, 
            'deg_exp':1.0, 
            'temp_dir_ID':0, 
            'full_path':None,
            'reservoir_weight_scale': 1.0,
            'input_weight_bounds': (0.0, 1.0),
            'loop_unique_input': False,
            'lower_reservoir_bound': 0.0,
            'upper_reservoir_bound': 1.0,
            'recall_task':False,
            'fp_convergence_delay':10,
            'fp_epsilon':0.001,
            'fp_distance_thresh':0.1}

        for key in kwargs.keys():
            if key not in parameter_defaults.keys():
                raise KeyError(key + " not a valid key")

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.num_trials = int(self.num_trials)
        self.reservoir_weight_bounds = np.array([self.lower_reservoir_bound, self.upper_reservoir_bound])

        # Create model and task
        self.esn = self.generate_esn()

        if self.distraction_range == None:
            self.task = Nbit_memory_task.Nbit_memory_task(distractor_value=self.distractor_value, 
                sequence_dimension=self.sequence_dimension,
                start_time=self.start_time,
                sequence_length=self.sequence_length,
                distraction_duration=self.distraction_duration,
                loop_unique_input=self.loop_unique_input, cue_value=self.cue_value)

        else:
            self.task = [Nbit_memory_task.Nbit_memory_task(distractor_value=self.distractor_value, 
                sequence_dimension=self.sequence_dimension,
                start_time=self.start_time,
                sequence_length=self.sequence_length, cue_value=self.cue_value,
                distraction_duration=np.random.randint(self.distraction_range[0], self.distraction_range[1])) 
                    for i in range(self.num_trials)]

    def generator_reservoir(self):
        """
        Use another function to create the reservoir and return it
        """

        if self.com_size != None:
            minc = self.com_size
            maxc = self.com_size
        else:
            minc = self.minc
            maxc = self.maxc
        if self.homok != None:
            k = self.homok
            maxk = self.homok
        else:
            k = self.k
            maxk = self.maxk
        
        self.nx_reservoir = graphgen.uniform_weighted_directed_lfr_graph(N=self.N, mu=self.mu, 
            k=k, maxk=maxk, minc=minc, maxc=maxc, deg_exp=self.deg_exp, 
            temp_dir_ID=self.temp_dir_ID, full_path=self.full_path, 
            weight_bounds=self.reservoir_weight_scale * self.reservoir_weight_bounds)

        return np.asarray(nx.to_numpy_matrix(self.nx_reservoir))

    def generate_input_weights(self):
        """
        Generates the input weights for the reservoir. It does this by generating first an array
        of either random or unitary array and then applying a mask to reduce the inputs down to
        that required by the input fraction. If a sequence of input gains are given, then the
        corresponding dimension's weights are adjusted by that gain, else if a scalar is given
        the whole array is adjusted.
        """

        input_weights = np.random.uniform(self.input_weight_bounds[0], self.input_weight_bounds[1], \
            size=(self.N, self.sequence_dimension + 2)) # 2 added for the distractor and cue
        
        if isinstance(self.input_gain, Sequence):
            for i in input_weights.shape[1]:
                input_weights[:,i] *= self.input_gain[i]
        else:
            input_weights *= self.input_gain

        fraction_mask = np.zeros(self.N * (self.sequence_dimension + 2))
        fraction_mask[:int(self.input_fraction * self.N)] = 1.0
        np.random.shuffle(fraction_mask)
        fraction_mask = fraction_mask.reshape((self.N, self.sequence_dimension + 2))
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

    def generate_recall_task_sequences(self):
        """
        Returns a set of unique task sequences for memorization
        """

        self.multitrial_input_signal = np.zeros((self.num_trials, self.task.total_duration, self.task.input_dimension, 1))
        self.multitrial_target_signal = np.zeros((self.num_trials, self.sequence_length, self.sequence_dimension, 1))
        self.task_sequences = set([])
        for trial in range(self.num_trials):
            input_signal, target_signal = self.task.generate_signal()
            check_input = tuple(input_signal.reshape(input_signal.shape[0] * input_signal.shape[1]))
            while check_input in self.task_sequences:
                input_signal, target_signal = self.task.generate_signal()
                check_input = tuple(input_signal.reshape(input_signal.shape[0] * input_signal.shape[1]))
            
            self.task_sequences.add(check_input)

            # reshape input into be compatible with ESN
            input_signal = input_signal.reshape((input_signal.shape[0], input_signal.shape[1], 1))
            target_signal = target_signal.reshape(target_signal.shape[0], target_signal.shape[1], 1)
            
            self.multitrial_input_signal[trial,:,:,:] = input_signal.copy()
            self.multitrial_target_signal[trial,:,:,:] = target_signal.copy()

        return self.multitrial_input_signal, self.multitrial_target_signal

    def generate_input(self, trials=None):
        """
        returns two arrays of size Q x T x K x 1 (num_trials, time, input dimension, 1[for vector operations])
        in second array (the target array) K=output dimension
        """

        if self.distraction_range == None:

            if trials == None:
                trials = self.num_trials

            multitrial_input_signal = np.zeros((trials, self.task.total_duration, self.task.input_dimension, 1))
            multitrial_target_signal = np.zeros((trials, self.sequence_length, self.sequence_dimension, 1))
            for trial in xrange(trials):
                input_signal, target_signal = self.task.generate_signal()
                # reshape input into be compatible with ESN
                input_signal = input_signal.reshape((input_signal.shape[0], input_signal.shape[1], 1))
                target_signal = target_signal.reshape(target_signal.shape[0], target_signal.shape[1], 1)
                
                multitrial_input_signal[trial,:,:,:] = input_signal.copy()
                multitrial_target_signal[trial,:,:,:] = target_signal.copy()

            return multitrial_input_signal, multitrial_target_signal

        else:

            if trials == None:
                trials = self.num_trials

            multitrial_input_signal = []
            multitrial_target_signal = []
            for i in xrange(trials):
                input_signal, target_signal = self.task[i].generate_signal()
                # reshape input into be compatible with ESN
                input_signal = input_signal.reshape((input_signal.shape[0], input_signal.shape[1], 1))
                target_signal = target_signal.reshape(target_signal.shape[0], target_signal.shape[1], 1)

                multitrial_input_signal.append(input_signal.copy())
                multitrial_target_signal.append(target_signal.copy())

            return multitrial_input_signal, multitrial_target_signal

    def TrainModel(self):
        """
        """

        if self.distraction_range == None:
            recall_times = [ self.task.recall_time for i in range(self.num_trials) ]
        else:
            recall_times = [ self.task[i].recall_time for i in range(self.num_trials) ]

        if self.recall_task:
            self.esn.MultiTrialTraining(*self.generate_recall_task_sequences(), cuts=0, recall_times=recall_times, cut_target_output=False)
            return None

        self.esn.MultiTrialTraining(*self.generate_input(), cuts=0, recall_times=recall_times, cut_target_output=False)

    def ValidateModel(self, num_trials=1):
        """
        Return a tuple containing the mean NRMSE and the fraction of correctly remembered sequences
        """

        if self.recall_task:
            test_input, test_targets = self.multitrial_input_signal, self.multitrial_target_signal
            num_trials = self.num_trials
        else:
            test_input, test_targets = self.generate_input(num_trials)

        if self.distraction_range == None:
            recall = [ self.task.recall_time for i in xrange(num_trials)]
        else:
            recall = [ self.task[i].recall_time for i in xrange(num_trials)]

        listNRMSE = []
        for i in xrange(num_trials):
            listNRMSE.append(self.esn.Predict(test_input[i], test_targets[i], 
                cut=0, recall_time=recall[i], cut_target_output=False, target_range=(0,1), error_type='NRMSE'))
            self.esn.Reset()

        return np.mean(listNRMSE), 1.0 - np.count_nonzero(listNRMSE) / float(len(listNRMSE))

    def is_state_changed(self, last_state, current_state):

        for i in range(self.esn.num_neurons):
            if last_state[i] != current_state[i]:
                return True

        return False

    def fixed_point_analysis(self):
        """
        Applies a fixed point analysis on the reservoir
        """

        stable_state_list = []
        num_diverged = 0.0
        input_signal, _ = self.generate_recall_task_sequences()
        for i in range(self.num_trials):

            # Run reservoir
            self.esn.Reset()
            # Loop up to but don't include the que and recall period.
            prev_state = self.esn.current_state
            max_iter = input_signal.shape[1]-(1+self.sequence_length)
            for j in range(max_iter):
                self.esn.Step(input_signal[i,j], record=True)
                if not self.is_state_changed(prev_state, self.esn.current_state):
                    break
                prev_state = self.esn.current_state

            # Check for stable points
            # if np.any(np.abs(self.esn.network_history[-1] - 
            #     np.array(self.esn.network_history[-self.fp_convergence_delay:-2])) > self.fp_epsilon):
            #             num_diverged += 1
            if j == (max_iter-1):
                num_diverged += 1
            else:
                # If it didn't diverge, then add it to the stable list
                stable_state_list.append(np.reshape(self.esn.network_history[-1], (self.esn.num_neurons)))

        unique_state_list = []
        while stable_state_list:
            for state in unique_state_list:
                if np.linalg.norm(stable_state_list[-1] - state) < self.fp_distance_thresh:
                    stable_state_list.pop()
                    break
            
            else:
                unique_state_list.append(stable_state_list.pop())

        return len(unique_state_list), num_diverged

if __name__ == '__main__':
    """
    """
    np.set_printoptions(suppress=True)
    nbit_obj = Nbit_memory_objective(mu=0.5, num_trials=100, N=1000, sequence_dimension=4, 
        neuron_type='tanh', #neuron_pars={'c':1,'e':10},
        input_weight_bounds=(-1.0,1.0), lower_reservoir_bound=-0.1, upper_reservoir_bound=1, 
        sequence_length=5, k=7, maxk=7, input_fraction=0.3,
        distraction_duration=80, distraction_range=None, distractor_value=0, 
        loop_unique_input=False, minc=10, maxc=10, input_gain=1.0, recall_task=True)
    # out, tar = nbit_obj.generate_input(1)
    # nbit_obj.esn.Train(out[0], tar[0], cut=0, recall_time=nbit_obj.task.recall_time, cut_target_output=False)
    # out, tar = nbit_obj.generate_input(1)
    # print nbit_obj.esn.Predict(out[0], tar[0], cut=0, recall_time=nbit_obj.task.recall_time, 
    #     cut_target_output=False, target_range=(0,1), error_type='NRMSE')
    nbit_obj.esn.Reset()
    nbit_obj.TrainModel()
    print nbit_obj.ValidateModel(100)