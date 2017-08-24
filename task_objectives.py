import networkx as nx
import graphgen
import echostatenetwork
import tasks
import numpy as np
from collections import Sequence

class base_objective(object):

    def __init__(self, **kwargs):

        parameter_defaults = {
            'target_range': (0,1),
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
            'reservoir_weight_bounds': (0.0, 1.0),
            'input_weight_bounds': (0.0, 1.0)}

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

    def generate_esn(self):
        """
        """

        self.reservoir = self.generator_reservoir()

        self.input_weights = self.generate_input_weights()
        return echostatenetwork.DESN(self.reservoir, self.input_weights, 
            neuron_type=self.neuron_type, output_type=self.output_neuron_type, 
            neuron_pars=self.neuron_pars, output_neuron_pars=self.output_neuron_pars,
            init_state="zeros")

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
            weight_bounds=self.reservoir_weight_scale * np.array(self.reservoir_weight_bounds))

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
            size=(self.N, self.task.get_input_dimensions()[1])) # 2 added for the distractor and cue
        
        if isinstance(self.input_gain, Sequence):
            for i in input_weights.shape[1]:
                input_weights[:,i] *= self.input_gain[i]
        else:
            input_weights *= self.input_gain

        fraction_mask = np.zeros(self.N * self.task.get_input_dimensions()[1])
        fraction_mask[:int(self.input_fraction * self.N)] = 1.0
        np.random.shuffle(fraction_mask)
        fraction_mask = fraction_mask.reshape((self.N, self.task.get_input_dimensions()[1]))
        input_weights = input_weights * fraction_mask

        return input_weights

    def generate_input(self, trials=None):
        """
        returns two arrays of size Q x T x K x 1 (num_trials, time, input dimension, 1[for vector operations])
        in second array (the target array) K=output dimension
        """

        if trials == None:
            trials = self.num_trials

        multitrial_input_signal = np.zeros((trials, self.task.get_input_dimensions()[0], self.task.get_input_dimensions()[1], 1))
        multitrial_target_signal = np.zeros((trials, self.task.get_output_dimensions()[0], self.task.get_output_dimensions()[1], 1))
        for trial in xrange(trials):
            input_signal, target_signal = self.task.generate_signal()
            # reshape input into be compatible with ESN
            input_signal = input_signal.reshape((input_signal.shape[0], input_signal.shape[1], 1))
            target_signal = target_signal.reshape(target_signal.shape[0], target_signal.shape[1], 1)
            
            multitrial_input_signal[trial,:,:,:] = input_signal.copy()
            multitrial_target_signal[trial,:,:,:] = target_signal.copy()

        return multitrial_input_signal, multitrial_target_signal

    def TrainModel(self):
        """
        """

        recall_times = [ self.task.recall_time for i in range(self.num_trials) ]
      
        self.esn.MultiTrialTraining(*self.generate_input(), cuts=0, recall_times=recall_times, cut_target_output=False)

    def ValidateModel(self, num_trials=1):
        """
        """

        test_input, test_targets = self.generate_input(num_trials)

        recall = [ self.task.recall_time for i in xrange(num_trials)]
        
        listNRMSE = []
        for i in xrange(num_trials):
            listNRMSE.append(self.esn.Predict(test_input[i], test_targets[i], 
                cut=0, recall_time=recall[i], cut_target_output=False, target_range=self.target_range, error_type='NRMSE'))
            self.esn.Reset()

        return np.mean(listNRMSE)

class null_objective(base_objective):

    def __init__(self, **kwargs):

        super(null_objective, self).__init__(**kwargs)

        task_defaults = {
            'sequence_dimension': 2,
            'sequence_length': 3,
            'normalized_input': True,
            'duration': 100
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.task = tasks.null_task(**self.__dict__)
        self.esn = self.generate_esn()

    def TrainModel(self):
        """dummy function for the null task"""
        pass

    def ValidateModel(self, num_trials=1):
        """
        dummy function for the null task
        This function runs the model for the desired number of trials
        and returns the activation output of the reservoir
        """

        multitrial_input, _dump = self.generate_input()
        total_activities = []
        fps = []
        haulting_times = []
        for i in range(multitrial_input.shape[0]):
            for j in xrange(multitrial_input[i].shape[0]):
                self.esn.Step(multitrial_input[i][j], record=True)
            total_activities.append(self.net_activity(self.esn.network_history))
            haulting_times.append(self.net_halting_time(self.esn.network_history))
            fps.append(self.last_activity(self.esn.network_history))
            self.esn.Reset()

        return np.mean(total_activities), np.mean(fps), np.mean(haulting_times)

    def net_activity(self, state_history):
        """
        """

        return np.sum(np.abs(state_history))

    def last_activity(self, state_history):
        """
        Returns last value taken by model
        """

        return np.sum(np.abs(state_history[-1]))

    def net_halting_time(self, state_history, low_bound = 0.001):
        """
        """

        for t, state in reversed(list(enumerate(state_history))):
            if (np.sum(np.abs(state)) / float(self.N)) > low_bound:
                return t

        return 0

class binary_memory_objective(base_objective):

    def __init__(self, **kwargs):

        super(binary_memory_objective, self).__init__(**kwargs)

        task_defaults = {
            'sequence_dimension' : 2,
            'sequence_length': 3,
            'normalized_input': True,
            'distraction_duration': 1,
            'distractor_value': 0,
            'output_neuron_type': 'heaviside',
            'output_neuron_pars': {'threshold': 0.5}
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.task = tasks.binary_memory_task(**self.__dict__)
        self.esn = self.generate_esn()

class poly_objective(base_objective):

    def __init__(self, **kwargs):

        super(poly_objective, self).__init__(**kwargs)

        task_defaults = {
            'sequence_dimension': 2,
            'sequence_length': 3,
            'exponent_sequence': (1,1),
            'normalized_input': True,
            'output_neuron_type': 'identity',
            'output_neuron_pars': {}
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.task = tasks.poly_task(**self.__dict__)
        self.esn = self.generate_esn()

class find_largest_objective(base_objective):

    def __init__(self, **kwargs):

        super(find_largest_objective, self).__init__(**kwargs)

        task_defaults = {
            'distraction_duration': 0,
            'sequence_dimension': 2,
            'sequence_length': 3,
            'exponent_sequence': (1,1),
            'normalized_input': True,
            'output_neuron_type': 'identity',
            'output_neuron_pars': {}
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.task = tasks.find_largest_task(**self.__dict__)
        self.esn = self.generate_esn()

if __name__ == '__main__':
    """testing"""

    np.set_printoptions(suppress=True)

    # # Test binary memory objective
    # test_obj =  binary_memory_objective(num_trials=200, N=500, sequence_dimension=3,
    #     reservoir_weight_bounds=(0.0,1.0), input_weight_bounds=(0.0,1.0), sequence_length=5, 
    #     distraction_duration=2)
    # # out, tar = test_obj.generate_input(1)
    # # test_obj.esn.Train(out[0], tar[0], cut=0, recall_time=test_obj.task.recall_time, cut_target_output=False)
    # test_obj.TrainModel()
    # print test_obj.ValidateModel(100)
    # # test_obj.esn.Reset()
    # # out, tar = test_obj.generate_input(1)
    # # print 'tar', tar
    # # print test_obj.esn.Predict(out[0], tar[0], cut=0, recall_time=test_obj.task.recall_time, 
    # #     cut_target_output=False, target_range=(0,1), error_type='NRMSE', analysis_mode=True)

    # # Test poly objective
    # test_obj = poly_objective(num_trials=2000, N=100, sequence_dimension=3, exponent_sequence=(2,1,3),
    #     reservoir_weight_bounds=(-1.0,1.0), input_weight_bounds=(-1.0,1.0), sequence_length=1000)
    # # out, tar = test_obj.generate_input(1)
    # # test_obj.esn.Train(out[0], tar[0], cut=0, recall_time=test_obj.task.recall_time, cut_target_output=False)
    # test_obj.TrainModel()
    # print test_obj.ValidateModel(1000)
    # # test_obj.esn.Reset()
    # # out, tar = test_obj.generate_input(1)
    # # print 'tar', tar
    # # print test_obj.esn.Predict(out[0], tar[0], cut=0, recall_time=test_obj.task.recall_time, 
    # #     cut_target_output=False, target_range=(0,1), error_type='NRMSE', analysis_mode=True)


    # # Test find largest objective
    # test_obj = find_largest_objective(num_trials=2000, N=50, sequence_dimension=3, exponent_sequence=(2,1,1),
    #     reservoir_weight_bounds=(-1.0,1.0), input_weight_bounds=(-1.0,1.0), sequence_length=5, distraction_duration=0)
    # # out, tar = test_obj.generate_input(1)
    # # test_obj.esn.Train(out[0], tar[0], cut=0, recall_time=test_obj.task.recall_time, cut_target_output=False)
    # test_obj.TrainModel()
    # print test_obj.ValidateModel(1000)
    # test_obj.esn.Reset()
    # out, tar = test_obj.generate_input(1)
    # print 'tar', tar
    # print 'out', out
    # print test_obj.esn.Predict(out[0], tar[0], cut=0, recall_time=test_obj.task.recall_time, 
    #     cut_target_output=False, target_range=(0,1), error_type='NRMSE', analysis_mode=True)

    # Test null objective
    test_obj = null_objective(num_trials=10, N=50, sequence_dimension=3,
        reservoir_weight_bounds=(-1.0,1.0), input_weight_bounds=(-1.0,1.0), sequence_length=5, duration=10)
    print test_obj.ValidateModel()

    pass