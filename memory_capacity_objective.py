import networkx as nx
import graphgen
import echostatenetwork
import numpy as np
from collections import Sequence

class memory_capacity_objective(object):

    def __init__(self, **kwargs):
        """
        
        Task parameters:
        sequence_length
        cut
        max_delay
        input_range

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
        input_weight_bounds
        lower_reservoir_bound
        upper_reservoir_bound

        """

        parameter_defaults = {
            'sequence_dimension': 1,
            'binary': False,
            'training_length': 10,
            'validation_length': 10,
            'cut': 0,
            'max_delay': 100,
            'shift': 1,
            'input_range': (-0.5, 0.5),
            'input_fraction': 0.2,
            'input_gain': 1.0,
            'neuron_type': 'sigmoid',
            'neuron_pars': {}, 
            'output_neuron_type': 'identity',
            'output_neuron_pars': {},
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
            'premade_esn': None,
            'target_spectral_radius': None}

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        if self.premade_esn == None:
            self.esn = self.generate_esn()
        else:
            self.esn = self.premade_esn['esn']
            self.reservoir = self.premade_esn['reservoir']
            self.nx_reservoir = self.premade_esn['nx_reservoir']
            self.input_weights = self.premade_esn['input_weights']

    def generator_reservoir(self):
        """
        Use another function to create the reservoir and return it
        """

        self.nx_reservoir = graphgen.uniform_weighted_directed_lfr_graph(N=self.N, mu=self.mu, 
            k=self.k, maxk=self.maxk, minc=self.minc, maxc=self.maxc, deg_exp=self.deg_exp, 
            temp_dir_ID=self.temp_dir_ID, full_path=self.full_path, 
            weight_bounds=self.reservoir_weight_scale * np.array([self.lower_reservoir_bound, self.upper_reservoir_bound]))

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
            size=(self.N, self.sequence_dimension))
        
        if isinstance(self.input_gain, Sequence):
            for i in input_weights.shape[1]:
                input_weights[:,i] *= self.input_gain[i]
        else:
            input_weights *= self.input_gain

        fraction_mask = np.zeros(self.N * self.sequence_dimension)
        fraction_mask[:int(self.input_fraction * self.N)] = 1.0
        np.random.shuffle(fraction_mask)
        fraction_mask = fraction_mask.reshape((self.N, self.sequence_dimension))
        input_weights = input_weights * fraction_mask

        return input_weights

    def generate_esn(self):
        """
        """

        self.reservoir = self.generator_reservoir()

        # Adjust spectral radius if not None
        if self.target_spectral_radius != None:
            spectral_radius = np.sort(np.absolute(np.linalg.eigvals(self.reservoir)))[-1]
            self.reservoir = self.reservoir / spectral_radius * self.target_spectral_radius

        self.input_weights = self.generate_input_weights()
        return echostatenetwork.DESN(self.reservoir, self.input_weights, 
            neuron_type=self.neuron_type, output_type=self.output_neuron_type, 
            neuron_pars=self.neuron_pars, output_neuron_pars=self.output_neuron_pars,
            init_state="zeros")

    def generate_signal(self, evaluation_length):
        """
        input signal is: T x 1
        target is: T x int(max_delay / shift)
        """

        self.sequence_length = self.cut + evaluation_length + self.max_delay
        if self.binary:
            input_signal = np.random.randint(2, size=(self.sequence_length, 1))
        else:
            input_signal = np.random.uniform(self.input_range[0], self.input_range[1], size=(self.sequence_length,1))

        target_signal = np.zeros((evaluation_length, int(self.max_delay / self.shift)))
        for i in range(target_signal.shape[1]):
            target_signal[:,i] = input_signal[self.cut + (i+1)*self.shift : evaluation_length + self.cut + (i+1)*self.shift, 0]

        return input_signal, target_signal

    def generate_input(self, evaluation_length):
        """
        Adjust shape for ESN by putting it in columnar format
        """

        input_signal, target_signal = self.generate_signal(evaluation_length)
        # reshape input into be compatible with ESN
        input_signal = input_signal.reshape((input_signal.shape[0], input_signal.shape[1], 1))
        target_signal = target_signal.reshape((target_signal.shape[0], target_signal.shape[1], 1))

        return input_signal, target_signal

    def train(self):
        """
        """

        self.esn.Train(*self.generate_input(self.training_length), 
            cut=self.cut+self.max_delay, cut_target_output=False)

    def validate(self):
        """
        """

        # Run validation test
        test_input, test_targets = self.generate_input(self.validation_length)
        self.esn.Reset()
        _, prediction, test_targets, _ = self.esn.Predict(test_input, test_targets, cut=self.cut+self.max_delay, 
            cut_target_output=False, error_type=None, analysis_mode=True)
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

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    mc_obj = memory_capacity_objective(training_length=500, validation_length=500,
        cut = 100, max_delay = 25, shift=1, binary=True, reservoir_weight_scale=44,
        input_fraction=0.3, input_gain=0.1, output_neuron_type='heaviside',
        output_neuron_pars={'threshold': 0.5}, input_range= (0.0, 1.0),
        neuron_type='sigmoid', neuron_pars={'c':1, 'e':10},
        mu=0.5, N=500, input_weight_bounds=(1.0,1.0), lower_reservoir_bound=0.0, 
        upper_reservoir_bound=1.0, k=6, maxk=6, minc=10, maxc=10)
    mc_obj.train()
    print mc_obj.validate()