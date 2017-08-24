import networkx as nx
import graphgen
import echostatenetwork
import numpy as np
from collections import Sequence

class SeparabilityAnalyzer(object):
    """
    Uses a normalized hamming distance as a metric for the distance
    between binary input arrays
    """

    def __init__(self, **kwargs):

        parameter_defaults = {
        'init_state': "zeros",
        'hamming_distance_range': np.linspace(0.1, 1., 10),
        'binomial_rate': 0.5,
        'sequence_length': 1000,
        'cut_length': 0,
        'num_trials_per_distance': 200,
        'input_fraction': 0.3,
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

        self.num_trials_per_distance = int(self.num_trials_per_distance)
        self.reservoir_weight_bounds = np.array([self.lower_reservoir_bound, self.upper_reservoir_bound])
        self.input_dimension = 1

        # Get euclidean distance range
        self.euclidean_range = [ self.get_euclidean_distance(hamming) for hamming in self.hamming_distance_range ]

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

    def find_neighbor_input(self, vector, distance):

       indexes_to_switch = np.random.choice(range(self.sequence_length), 
            size=int(distance * self.sequence_length), replace=False)
       neighbor = vector.copy()
       neighbor[indexes_to_switch] = 1 - neighbor[indexes_to_switch]
       return neighbor

    def create_input_pair(self, distance):

        input_sequence1 = np.random.binomial(1 , self.binomial_rate, size=(self.sequence_length, 1, 1))
        return input_sequence1, self.find_neighbor_input(input_sequence1, distance)

    def create_washout_input(self):

        return np.random.binomial(1, self.binomial_rate, size=(self.cut_length, 1, 1))

    def evaluate_esn_state_differences(self, distance):

        washout_input = self.create_washout_input()
        input_sequence, neighbor_sequence = self.create_input_pair(distance)
        input_sequence = np.concatenate((washout_input, input_sequence))
        neighbor_sequence = np.concatenate((washout_input, neighbor_sequence))

        # Run first input sequence
        self.esn.Reset()
        for i in range(self.sequence_length + self.cut_length):
            self.esn.Step(input_sequence[i])
        final_state_1 = self.esn.current_state

        # Run second input sequences
        self.esn.Reset()
        for i in range(self.sequence_length + self.cut_length):
            self.esn.Step(neighbor_sequence[i])
        final_state_2 = self.esn.current_state

        return np.linalg.norm(final_state_1 - final_state_2)

    def run_trials_at_given_distance(self, distance):

        trial_separations = []
        for i in range(self.num_trials_per_distance):
            trial_separations.append(self.evaluate_esn_state_differences(distance))

        return trial_separations

    def evaluate_trials_over_all_distances(self):

        separations_by_distance_by_trial = []
        for distance in self.hamming_distance_range:
            trial_separations = self.run_trials_at_given_distance(distance)
            separations_by_distance_by_trial.append(trial_separations)

        return np.array(separations_by_distance_by_trial)

    def get_euclidean_distance(self, hamming_distance):

        a, b = self.create_input_pair(hamming_distance)
        return np.linalg.norm(a-b)

if __name__ == '__main__':
    """
    for testing
    """

    test_analyzer = SeparabilityAnalyzer(**{
        'init_state': "zeros",
        'hamming_distance_range': np.linspace(0.0, 1., 10),
        'binomial_rate': 0.5,
        'sequence_length': 2000,
        'cut_length': 0,
        'num_trials_per_distance': 10,
        'input_fraction': 0.3,
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
    separations_by_distance_by_trial = test_analyzer.evaluate_trials_over_all_distances()

    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    plt.plot(test_analyzer.euclidean_range, np.mean(separations_by_distance_by_trial, axis=1))
    plt.subplot(2,1,2)
    plt.plot(test_analyzer.hamming_distance_range, np.mean(separations_by_distance_by_trial, axis=1))
    plt.show()