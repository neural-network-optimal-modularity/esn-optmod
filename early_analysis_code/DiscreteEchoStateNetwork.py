# Nathaniel Rodriguez

import numpy as np 
from numpy import linalg
import math
import sys
from functools import partial
import scipy.stats as stats

class ESN(object):
    """
    This is a discrete non-feedback ESN that uses linear regression learning.
    It can be expanded to include feedback, but this would require using a batch or online learning algorithm.
    Note: According to Jaeger, feedback is only needed for pattern generating ESNs. For time-series prediction, control,
    filtering, or pattern recognition, feedback is not advised.
    """

    def __init__(self, reservoir, input_weights, neuron_type="tanh", 
        output_type="sigmoid", init_state="zeros", neuron_pars={}, output_neuron_pars={}):
        """
        reservoir - NxN numpy array with weights for connected nodes
        input_weights - NxK numpy array with input signals
        """

        # Weights
        self.reservoir = reservoir
        self.num_neurons = self.reservoir.shape[0]
        self.input_weights = input_weights

        # Set neuron types (reservoir)
        self.neuron_pars = neuron_pars
        self.neuron_type = neuron_type
        if self.neuron_type == "tanh":
            self.activation_function = self.tanh
        elif self.neuron_type == "sigmoid":
            self.activation_function = partial(self.sigmoid, **neuron_pars)
        elif self.neuron_type == "RLU":
            self.activation_function = partial(self.rectified_linear_unit, **neuron_pars)
        elif self.neuron_type == "heaviside":
            self.activation_function = partial(self.heaviside, **neuron_pars)
        # Set neuron types (output neuron)
        self.output_type = output_type
        if self.output_type == "tanh":
            self.output_function = partial(self.tanh, **output_neuron_pars)
        elif self.output_type == "sigmoid":
            self.output_function = partial(self.sigmoid, **output_neuron_pars)
        elif self.output_type == "identity":
            self.output_function = partial(self.identity, **output_neuron_pars)
        elif self.output_type == "heaviside":
            self.output_function = partial(self.heaviside, **output_neuron_pars)

        # Generate initial system state
        self.init_state = init_state
        self.current_state = self.GenerateInitialState(self.init_state)
        self.network_history = []

    def Noise(self):

        return self.weiner_var * np.random.normal(loc=0.0, scale=1.0, size=(self.num_neurons,1))

    def GenerateInitialState(self, setup="zeros"):
        """
        Sets all initial states 
        """

        if setup == "zeros":
            return np.zeros((self.num_neurons, 1))
        else:
            return np.random.uniform(setup[0],setup[1],size=(self.num_neurons, 1))

    def Reset(self):

        self.network_history = []
        self.current_state = self.GenerateInitialState(self.init_state)

    def Step(self, input_vector, record=False):
        """
        input_vector - numpy array of dimensions Kx1
        """
        self.current_state = self.activation_function(np.dot(self.input_weights, input_vector) + \
                                np.dot(self.reservoir , self.current_state) )
        if record:
            self.network_history.append(self.current_state)

    def Response(self, input_vector, record=False):
        """
        Calculate the networks repsonse given an input vector
        """
        self.Step(input_vector, record)
        return self.output_function(np.dot(np.transpose(self.output_weight_matrix), 
            np.concatenate((self.current_state, input_vector), axis=0)))

    def RunModel(self, input_time_series, record=False):

        # Evaluate for time series input
        model_output = []
        for i in xrange(input_time_series.shape[0]):
            model_output.append(self.Response(input_time_series[i], record))

        return np.array(model_output)

    def sigmoid(self, x, a=1.0, b=1.0, c=0.0, d=0.0, e=1.0):
        """
        numpy Vector/matrix friendly sigmoid function
        """
        return a / (b + np.exp(-e*(x-c))) + d

    def inv_sigmoid(self, x, a=1.0, b=1.0, c=0.0, d=0.0, e=1.0):

        return - np.log(a / (x-d) - b) / e + c

    def tanh(self, x):
        """
        """
        return np.tanh(x)

    def identity(self, x):
        """
        """
        return x

    def rectified_linear_unit(self, x, threshold=0.0, scale=1.0):
        """
        """

        return scale * stats.threshold(x, threshmin=threshold, newval=0)

    def heaviside(self, x, threshold=0.0, newval=1.0):
        """
        """

        return newval * (x > threshold)

    def output_function_signal_map(target):
        """
        """

        if self.output_type == 'sigmoid':
            return inv_sigmoid(target, **self.output_neuron_pars)
        elif self.output_type == 'tanh':
            return np.arctanh(target)

    def SetOutputWeights(self, weight_matrix):
        """
        """

        self.output_weight_matrix = np.copy(weight_matrix)

    def MultiTrialTraining(self, array_input_time_series, array_target_output, cut, 
            recall_time=None, cut_target_output=True, invert_target=False):
        """
        array_input_time_series: ts for each trial (Q=num trials): Q x T x K x 1
        array_target_output: target output for each trial: Q x T x O x 1
        """

        if recall_time != None:
            cut = recall_time
        if invert_target:
            array_target_output = self.output_function_signal_map(array_target_output)

        relevant_history = np.zeros(((array_input_time_series.shape[1]-cut) * array_input_time_series.shape[0],
            array_input_time_series.shape[2] + self.current_state.shape[0]))
        for i in xrange(array_input_time_series.shape[0]):
            for j in xrange(array_input_time_series[i].shape[0]):
                self.Step(array_input_time_series[i][j], record=True)

            esn_ts = np.array(self.network_history)[cut:]
            cut_ts = array_input_time_series[i][cut:]
            extended_state = np.concatenate( (esn_ts, cut_ts), axis=1)
            extended_state =  np.reshape(extended_state, (extended_state.shape[0], extended_state.shape[1]))
            relevant_history[i * (array_input_time_series.shape[1]-cut): (i+1) * (array_input_time_series.shape[1]-cut),:] \
                = extended_state.copy()
            self.Reset()

        S = np.asmatrix(relevant_history)
        if cut_target_output:
            stacked_target_output = np.zeros(((array_target_output.shape[1]-cut) * array_target_output.shape[0], \
                array_target_output.shape[2]))

            for i in xrange(array_target_output.shape[0]):
                target_output = array_target_output[i,cut:].copy() # Should be an t-cut x num_outputs matrix
                target_output = np.reshape(target_output, (target_output.shape[0], target_output.shape[1]))
                stacked_target_output[i * (array_target_output.shape[1]-cut): (i+1) * (array_target_output.shape[1]-cut),:] \
                    = target_output.copy()
        else:
            stacked_target_output = np.zeros(((array_target_output.shape[1]) * array_target_output.shape[0], \
                array_target_output.shape[2]))

            for i in xrange(array_target_output.shape[0]):
                target_output = array_target_output[i,:].copy() # Should be an t-cut x num_outputs matrix
                target_output = np.reshape(target_output, (target_output.shape[0], target_output.shape[1]))
                stacked_target_output[i * (array_target_output.shape[1]): (i+1) * (array_target_output.shape[1]),:] \
                    = target_output.copy()                

        D = np.asmatrix(stacked_target_output)
        solution, residuals, rank, sing = linalg.lstsq(S, D)
        self.SetOutputWeights( solution ) 

    def Train(self, input_time_series, target_output, cut, 
            recall_time=None, cut_target_output=True, invert_target=False):
        """
        input_time_series - of dimensions TxKx1
        target_output - of dimensions TxKx1
        """

        if invert_target:
            target_output = self.output_function_signal_map(target_output)

        # Run network for full time-series
        for i in xrange(input_time_series.shape[0]):
            self.Step(input_time_series[i], record=True)
        
        return self.EvaluateOutputWeights(input_time_series, target_output, cut, recall_time, cut_target_output)

    def EvaluateOutputWeights(self, input_time_series, target_output, cut, recall_time=None, cut_target_output=True):
        """
        input_time_series - of dimensions TxKx1
        target_output - of dimensions TxKx1

        or if recall is enabled: then cut output recording till the recall time
        """


        if recall_time != None:
            cut = recall_time

        # Construct S -> concatenation of reservoir states and input states
        esn_ts = np.array(self.network_history)[cut:]
        cut_ts = input_time_series[cut:]

        cut_extended_state_matrix = np.concatenate( (esn_ts, cut_ts), axis=1)
        cut_extended_state_matrix = np.reshape(cut_extended_state_matrix, (cut_extended_state_matrix.shape[0], cut_extended_state_matrix.shape[1]))
        S = np.asmatrix(cut_extended_state_matrix)
        # make sure outcome matrix is a matrix
        if cut_target_output:
            target_output = np.array(target_output[cut:]) # Should be an t-cut x num_outputs matrix
        target_output = np.reshape(target_output, (target_output.shape[0], target_output.shape[1]))
        # target_output
        D = np.asmatrix(target_output)
        solution, residuals, rank, sing = linalg.lstsq(S, D)
        self.SetOutputWeights( solution )

    def NormalizedRootMeanSquaredError(self, residuals, ymax, ymin):

        return np.sqrt(np.mean(np.power(residuals, 2))) / (ymax - ymin)

    def AbsoluteError(self, residuals):
        """
        """

        return np.sum(np.absolute(residuals))

    def Predict(self, input_time_series, target_output, cut, recall_time=None, cut_target_output=True,
        target_range=None, error_type='NRMSE', analysis_mode=False):

        prediction = self.RunModel(input_time_series, record=analysis_mode)

        if analysis_mode:
            full_output = prediction.copy()

        # print 'prediction', prediction.shape, prediction
        # Calculate NRMSE
        if recall_time != None:
            cut = recall_time
        # print "cut pred:", prediction[cut:]
        prediction = prediction[cut:]
        prediction = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
        # print 'prediction', prediction
        if cut_target_output:
            target_output = np.array(target_output[cut:]) # Should be an t-cut x num_outputs matrix
        target_output = np.reshape(target_output, (target_output.shape[0], target_output.shape[1]))
        # target_output = target_output[:,0]
        residuals = np.abs(prediction - target_output)

        if error_type == 'NRMSE':
            if target_range != None:
                min_target = target_range[0]
                max_target = target_range[1]
            else:
                min_target = np.min(target_output)
                max_target = np.max(target_output)

            performance =  self.NormalizedRootMeanSquaredError(residuals, max_target, min_target)
        elif error_type == 'AE':
            performance =  self.AbsoluteError(residuals)

        if analysis_mode:
            return performance, prediction, target_output, full_output
        else:
            return performance

if __name__ == '__main__':
    """
    testing
    """

    pass
