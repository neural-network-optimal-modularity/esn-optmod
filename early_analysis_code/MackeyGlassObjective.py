"""
Author: Nathaniel Rodriguez

ESN time series prediction objective function

"""

import DiscreteEchoStateNetwork
import DiscreteMackeyGlass
import math
import random as rnd
import numpy as np
from numpy import linalg
import sys
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import itertools
import WeightedStochasticBlockModel as WSBM


class MackeyPrediction(object):

    def __init__(self, num_groups, N_range=(1,100), group_fraction_range=(0.0,1.0), degree_dist_shape_range=(0.0, 0.0), 
        degree_dist_scale_range=(0.0, 0.0), block_element_range=(0.001, 1.0),
        W_in_bias_range=(0.01,10), W_in_fraction_range=(0.01,1), 
        spectral_radius_range=(0.1,2), w_gamma_shape_range=(0.01,10), w_gamma_scale_range=(0.01,10), 
        beta=0.2, gamma=0.1, n=10.0, tau=17, training_time=5000,
        lag=5, cut=300, trials=1, unique_group_weights=False, random_input=False):

        # Glass parameters
        self.beta = beta
        self.gamma = gamma
        self.n = n 
        self.tau = tau

        # Training parameters
        self.training_time = training_time
        self.lag = lag
        self.cut = int(cut)
        self.trials = trials

        # genome parameters
        self.num_groups = num_groups
        self.unique_group_weights = unique_group_weights
        self.num_inputs = 1
        self.N_range = N_range
        self.group_fraction_range = group_fraction_range
        self.degree_dist_shape_range = degree_dist_shape_range
        self.degree_dist_scale_range = degree_dist_scale_range
        self.block_element_range = block_element_range
        self.W_in_bias_range = W_in_bias_range
        self.W_in_fraction_range = W_in_fraction_range
        self.spectral_radius_range = spectral_radius_range
        self.w_gamma_shape_range = w_gamma_shape_range
        self.w_gamma_scale_range = w_gamma_scale_range
        
        # WSBM parameters
        self.relative_group_sizes = np.zeros(self.num_groups)
        self.connectivity_block_matrix = np.zeros((self.num_groups, self.num_groups))
        self.weight_block_matrix = np.zeros((self.num_groups, self.num_groups, 3))
        self.random_input = random_input

    def GenerateESN(self):

        # Generate time constants
        self.time_constants = np.clip(np.random.gamma(shape=self.tc_gamma_shape, scale=self.tc_gamma_scale, size=(self.N,1)), 0.0001, 1000.0)

        # Construct reservoir
        self.reservoir = self.GenerateReservoir()

        # Construct input weights
        self.input_weights = self.GenerateInputWeightMatrix()

        # Construct ESN
        return DiscreteEchoStateNetwork.ESN(self.reservoir, self.input_weights, neuron_type="tanh", output_type="tanh")

    def GenerateDiscreteMackeyGlassTimeSeries(self):

        # Construct DiscreteMackeyGlass
        ic = np.random.uniform(0.0,1.0, size=int(self.tau) + 1)
        self.macky = DiscreteMackeyGlass.DiscreteMackeyGlassSystem(ic, self.beta, self.gamma, self.n, self.tau)

        # Generate time-series (squash between -1 and 1)
        time_series = self.tanh(np.array(self.macky.generate_series(int(self.training_time))) - 1.)
        time_series = time_series.reshape((len(time_series), 1, 1))

        # Generate target time-series (and adjust for lag)
        if self.lag >= 0:
            target_time_series = time_series[int(math.ceil(self.lag + 0.4):]
            time_series = time_series[:-int(math.ceil(self.lag) + 0.4)]
        else:
            target_time_series = time_series[:-int(math.ceil(-self.lag) + 0.4)]
            time_series = time_series[int(math.ceil(-self.lag) + 0.4):]

        return target_time_series, time_series

    def __call__(self, genome):

        # Convert the genome into model variables
        self.ConvertGenome(genome)

        fitnesses = []

        for i in xrange(self.trials):

            # Generate ESN
            esn = self.GenerateESN()

            # Construct system and carry-out preprocessing
            target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()

            # Train ESN
            error = esn.Train(time_series, target_time_series, self.cut)

            # Validation test
            target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()
            error = esn.Predict(time_series, target_time_series, self.cut)
            fitnesses.append(-1. * error)

        return np.median(fitnesses), np.array(fitnesses)

    def CalculateLargestEigenval(self, adj_matrix):
        try:
            eigenvalues = linalg.eigvals(adj_matrix)
        except linalg.LinAlgError:
            print "Error: Failure to converge matrix:", weight_scale_factor
            sys.exit()

        largest_eigenvalue = np.real(eigenvalues).max()

        return largest_eigenvalue

    def GenerateReservoir(self):

        # Create randomly connected directed ER graph with density p
        reservoir, self.group_membership = WSBM.WSBM(self.N, self.relative_group_sizes, \
            (self.degree_dist_shape, self.degree_dist_scale), \
            self.connectivity_block_matrix, self.weight_block_matrix, True)

        # Get ESN property indicator --------change for non leaky
        reservoir =  reservoir * np.identity(self.N)
        effective_spectral_radius = self.CalculateLargestEigenval(reservoir)
        reservoir = reservoir / effective_spectral_radius * self.spectral_radius

        # Remove self-connections
        np.fill_diagonal(reservoir, 0.0)

        return reservoir

    def convert_range(self, value, low, high):

        new_range = high - low
        return value * new_range + low

    def tanh(self, x):
        """
        """
        return np.tanh(x)

    def GenerateInputWeightMatrix(self):
        """
        Creates a randomly weighted [-1,1] input weight matrix of size N x num_inputs
        Has number of input connections = input_fraction * N (for each input)
        """

        input_weights = np.random.uniform(-1.0, 1.0, size=(self.N, 1))

        # Adjust input fraction
        if self.random_input:
            input_mask = np.ones((self.N,1))
        else:
            input_mask = np.array([ 1 if self.group_membership[i]==0 else 0 for i in xrange(self.N)], ndmin=2).T

        fraction_mask = np.random.choice([0,1], size=(self.N,1), p=[1 - self.W_in_fraction, self.W_in_fraction])
        input_weights = input_weights * fraction_mask * input_mask

        # Adjust bias
        input_weights = input_weights * self.W_in_bias

        return input_weights

    def Test(self, genome=None):

        if genome != None:
            self.ConvertGenome(genome)

        # print "N: ",self.N, \
        #         "p: ", self.p, \
        #         "W_in_bias: ", self.W_in_bias, \
        #         "W_in_fraction: ", self.W_in_fraction, \
        #         "Spectral radius: ", self.spectral_radius, \
        #         "W_gamma_shape: ", self.w_gamma_shape, \
        #         "W_gamma_scale: ", self.w_gamma_scale,\
        #         "leak_rate: ", self.leak_rate, \
        #         "dt: ", self.delta_t, \
        #         "tc_gamma_shape: ", self.tc_gamma_shape, \
        #         "tc_gamma_scale: ", self.tc_gamma_scale, \
        #         "noise_strength: ", self.noise_strength

        esn = self.GenerateESN()
        target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()
        # Train ESN
        error = esn.Train(time_series, target_time_series, self.cut)
        esn.Reset()

        # Reset glass and run
        target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()
        model_response = esn.RunModel(time_series)
        
        # Plot
        plt.plot(model_response[:,0,0], '--', color='blue', label='Model')
        plt.plot(time_series[:,0,0], '-', color='black', label='Signal')
        plt.plot(target_time_series[:,0,0], '--', color='red', label='Target')
        plt.legend()
        plt.show()
        # plt.savefig('time_series_plot.png',dpi=300)

if __name__ == '__main__':
    """
    testing
    """
    pass