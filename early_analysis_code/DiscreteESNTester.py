"""
Author: Nathaniel Rodriguez
Class is for testing the results of the ESN
"""

import matplotlib
# matplotlib.use('Agg')
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
from MackeyGlassObjective import *
import networkx as nx

class ESNTest(object):

    def __init__(self, genome, objective, prefix="test"):

        # Additional parameters
        self.genome = genome
        self.prefix = prefix
        self.objective_object = objective

        # Convert genome
        self.ConvertGenome(self.genome)

    def __call__(self, *args, **kwargs):
        return self.objective_object(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.objective_object, attr)

    # def __setattr__(self, x, attr, val):
    #     return setattr(self.objective_object, attr, val)

    def ReservoirPerformanceTest(self, trials):

        self.objective_object.trials = trials
        median_fitness, fitnesses = self(self.genome)

        # Plot performance distribution
        plot_ccdf(self.prefix + "_mackeyglass_fitness_distribution", fitnesses, xlabel="-NRMSE")
        # Also plot histogram for good measure
        plt.clf()
        plt.hist(fitnesses, bins=int(math.sqrt(len(fitnesses))))
        plt.xlabel("-NRMSE")
        plt.ylabel("PDF")
        plt.tight_layout()
        plt.savefig(self.prefix + "_mg_fitness_histogram.png", dpi=300)
        plt.close()
        plt.clf()
        print "\nMedian Error: ", np.median(fitnesses), '\n'

    def ReservoirLagPerformanceTest(self, delay_range, trials=30, num_delays=100):
        
        # Set attribute for trials
        self.objective_object.trials = trials

        # for each delay, run performance test:
        listDelays = []
        listMedianFitnessPerDelay = []
        listDelayslistFitnesses = []
        for i, delay in enumerate(np.linspace(delay_range[0], delay_range[1], num_delays)):
            print "\t", float(i) / num_delays
            median_fitness, fitnesses = self(self.genome)

            listDelays.append(delay)
            listDelayslistFitnesses.append(fitnesses)
            listMedianFitnessPerDelay.append(median_fitness)

        # Get quantiles for each delay
        upper_q_list = []
        med_list = []
        lower_q_list = []
        for fitness_list in listDelayslistFitnesses:
            low, med, high = np.percentile(fitness_list, [25,50,75])
            upper_q_list.append(high)
            med_list.append(med)
            lower_q_list.append(low)

        # Plot median performance per delay w/ 95% quantiles (shaded)
        plt.clf()
        plt.plot(listDelays, med_list, marker='o', ls='-', color='#CC4F1B', markersize=4)
        plt.ylabel('-NRMSE')
        plt.xlabel('$lag$')
        plt.axvline(self.lag,lw=2, ls='--',color='black')
        plt.fill_between(listDelays, lower_q_list, upper_q_list, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.prefix + '_fitness_vs_delay.png', dpi=300)
        plt.close()
        plt.clf()

    def NoRetrainingLagPerformanceTest(self, delay_range, trials=30, num_delays=100):

        # Set attribute for trials
        self.objective_object.trials = trials

        # For each trained ESN drawn from the distribution run delay tests
        listESNlistDelays = []
        listESNlistFitness = []
        for trial in xrange(trials):
            print "\t", trial
            # Generate ESN  
            esn = self.GenerateESN()

            # Construct system and carry-out preprocessing
            target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()

            # Train ESN
            print self.lag
            error = esn.Train(time_series, target_time_series, self.cut)

            # For each delay, test the trained ESN at prediction
            listDelays = []
            listFitness = []
            print np.linspace(delay_range[0], delay_range[1], num_delays)
            for delay in np.linspace(delay_range[0], delay_range[1], num_delays):

                # Generate Glass
                old_lag = self.lag
                self.lag = delay 
                print "\t", self.lag
                target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()
                self.lag = old_lag

                # Validate
                esn.Reset()
                error = esn.Predict(time_series, target_time_series, self.cut)
                listFitness.append(-1. * error)
                listDelays.append(delay)

            listESNlistDelays.append(listDelays)
            listESNlistFitness.append(listFitness)

        # Reshape results
        listDelayslistFitness = zip(*listESNlistFitness)
        listDelays = listESNlistDelays[0]

        # Get quantiles for each delay
        upper_q_list = []
        med_list = []
        lower_q_list = []
        for fitness_list in listDelayslistFitness:
            low, med, high = np.percentile(fitness_list, [25,50,75])
            upper_q_list.append(high)
            med_list.append(med)
            lower_q_list.append(low)

        # Plot median performance per delay w/ 95% quantiles (shaded)
        plt.clf()
        plt.plot(listDelays, med_list, marker='o', ls='-', color='#CC4F1B', markersize=4)
        plt.ylabel('-NRMSE')
        plt.xlabel('$lag$')
        plt.axvline(self.lag,lw=2, ls='--',color='black')
        plt.fill_between(listDelays, lower_q_list, upper_q_list, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.prefix + '_noretraining_fitness_vs_delay.png', dpi=300)
        plt.close()
        plt.clf()

    def plot_connectivity_block_matrix(self):

        plt.clf()
        plt.pcolor(self.connectivity_block_matrix, cmap="binary", vmin=0.0, vmax=1.0)
        # plt.colorbar()
        plt.xticks(np.arange(0.5,self.num_groups + 0.5),range(0,self.num_groups),fontsize=19)
        plt.yticks(np.arange(0.5,self.num_groups + 0.5),range(0,self.num_groups),fontsize=19)
        # plt.
        plt.gca().invert_yaxis()

        plt.savefig(self.prefix + "_block_matrix.png", dpi=300)
        plt.clf()
        plt.close()

    def generate_graph(self):

        graph = nx.DiGraph(self.GenerateReservoir())
        nx.write_pajek(graph, self.prefix + "_graph.net")
        for node in graph.nodes_iter():
            graph.node[node]['community'] = self.group_membership[node]

        nx.write_gexf(graph, self.prefix + "_graph.gexf")

    def Test(self):

        esn = self.GenerateESN()
        target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()
        # Train ESN
        error = esn.Train(time_series, target_time_series, self.cut)
        esn.Reset()

        # Reset glass and run
        target_time_series, time_series = self.GenerateDiscreteMackeyGlassTimeSeries()
        model_response = esn.RunModel(time_series)
        
        # Plot
        print "\nError:", error, '\n'
        plt.plot(model_response[:,0,0], '--', color='blue', label='Model')
        plt.plot(time_series[:,0,0], '-', color='black', label='Signal')
        plt.plot(target_time_series[:,0,0], '--', color='red', label='Target')
        plt.legend()
        plt.show()
        # plt.savefig(self.prefix + '_time_series_plot.png',dpi=300)

def ecdf(data):

    sorted_data = np.sort(data)
    size = float(len(sorted_data))
    cdf = np.array([ i / size for i in xrange(1,len(sorted_data)+1) ])
    return sorted_data, cdf

def eccdf(data):

    sorted_data, cdf = ecdf(data)
    return sorted_data, 1. - cdf

def plot_ccdf(prefix, data, xlabel='', x_log=False, y_log=False):

    x, y = eccdf(data)
    plt.clf()
    plt.plot(x, y, ls='-', marker='o', color='blue')
    if x_log == True: plt.xscale('log')
    if y_log == True: plt.yscale('log')
    plt.ylabel('CCDF')
    plt.xlabel(xlabel)
    plt.savefig(prefix + '.png', dpi=300)
    plt.clf()
    plt.close()