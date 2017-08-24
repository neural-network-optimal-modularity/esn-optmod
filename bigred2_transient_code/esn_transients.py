import networkx as nx
import graphgen
import echostatenetwork
import numpy as np
from collections import Sequence

class TransientAnalyzer(object):

    def __init__(self, **kwargs):

        parameter_defaults = {
        'init_state': "zeros",
        'max_iter': 1000,
        'num_trials': 100,
        'input_dimension': 1,
        'input_type': None,
        'input_params': {},
        'input_fraction': 0.1,
        'input_gain': 1.0,
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
        'lower_reservoir_bound': 0.0,
        'upper_reservoir_bound': 1.0
        }

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.num_trials = int(self.num_trials)
        self.reservoir_weight_bounds = np.array([self.lower_reservoir_bound, self.upper_reservoir_bound])

        # Create model and task
        self.esn = self.generate_esn()

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
            size=(self.N, self.input_dimension))
        
        if isinstance(self.input_gain, Sequence):
            for i in input_weights.shape[1]:
                input_weights[:,i] *= self.input_gain[i]
        else:
            input_weights *= self.input_gain

        fraction_mask = np.zeros(self.N * (self.input_dimension))
        fraction_mask[:int(self.input_fraction * self.N)] = 1.0
        np.random.shuffle(fraction_mask)
        fraction_mask = fraction_mask.reshape((self.N, self.input_dimension))
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
            init_state=self.init_state)

    def input_sequence(self):

        if self.input_type == None:
            return np.zeros((self.num_trials, self.max_iter, self.input_dimension, 1))

        elif self.input_type == 'impulse':
            return np.ones((self.num_trials, self.input_params['duration'], self.input_dimension, 1)) * self.input_params['scale']

        elif self.input_type == 'rand_impulse':
            return np.random.uniform(self.input_params['lower_bound'], self.input_params['upper_bound'],
             size=(self.num_trials, self.input_params['duration'], self.input_dimension, 1))

    def is_state_changed(self, last_state, current_state):

        for i in range(self.esn.num_neurons):
            if last_state[i] != current_state[i]:
                return True

        return False

    def evaluate_transient_length(self, trial_input):
        """
        evaluates the length of the transient up to max_iter
        for the reservoir for a single trial
        """

        self.esn.Reset()
        prev_state = self.esn.current_state
        for i in range(self.max_iter):
            self.esn.Step(trial_input[i])
            if not self.is_state_changed(prev_state, self.esn.current_state):
                return i

            prev_state = self.esn.current_state

        return i

    def evaluate_reservoir_transient_length_trials(self):
        """
        """

        inputs = self.input_sequence()
        finite_transient_lengths = []
        inf_transient_count = 0
        for i in range(self.num_trials):
            transient_length = self.evaluate_transient_length(inputs[i])
            if transient_length == (self.max_iter - 1):
                inf_transient_count += 1
                finite_transient_lengths.append(np.nan)
            else:
                finite_transient_lengths.append(transient_length)

        return finite_transient_lengths, inf_transient_count

if __name__ == '__main__':
    """
    for testing
    """

    test_analyzer = TransientAnalyzer(**{
        'init_state': (0.0, 1.0),
        'max_iter': 2000,
        'num_trials': 10,
        'input_dimension': 1,
        'input_type': 'impulse',
        'input_params': {'duration': 1, 'scale': 1},
        'input_fraction': 0.1,
        'input_gain': 1.0,
        'neuron_type': 'sigmoid',
        'neuron_pars': {'c':1, 'e':10}, 
        'output_neuron_type': 'heaviside',
        'output_neuron_pars': {'threshold': 0.5},
        'N': 500, 
        'mu': 0.25,
        'k': 6, 
        'maxk': 6,
        'homok': None,
        'com_size': None, 
        'minc':10, 
        'maxc':10, 
        'deg_exp':1.0, 
        'temp_dir_ID':0, 
        'full_path':None,
        'reservoir_weight_scale': 1.132,
        'input_weight_bounds': (0.0, 1.0),
        'lower_reservoir_bound': -0.2,
        'upper_reservoir_bound': 1.0
        })
    transient_lengths = test_analyzer.evaluate_reservoir_transient_length_trials()
    print transient_lengths