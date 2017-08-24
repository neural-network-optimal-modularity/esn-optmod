"""
Author: Nathaniel Rodriguez

Requirements:
-python (used 2.7.3)
-matplotlib (1.4.0 +)
-numpy (0.14.0 +)
-scipy (1.9.1 +)
-networkx (1.6 +) [only needed for making graphs of networks]
"""

import matplotlib
# matplotlib.use('Agg')
import MackeyGlassObjective
import DiscreteESNTester
import EvolveReservoir
import DiscreteEchoStateNetwork
import matplotlib.pyplot as plt
import sys
import os
import numpy as np 
import scipy.stats as stats
import copy
from multiprocessing import Pool
from functools import partial
import pickle
import random as rnd
import DiscreteMackeyGlass
import math

def save_object(obj, filename):
    """
    Save an object to file for later use.
    """
    
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()

def load_object(filename):
    """
    Load a previously saved object.
    """
    
    file = open(filename, 'rb')
    return pickle.load(file)

def parallel_function(f):

    def easy_parallize(f, sequence):
        """ assumes f takes sequence as input, easy w/ Python's scope """

        pool = Pool(processes=NUM_CPU) # depends on available cores
        result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
        cleaned = [x for x in result if not x is None] # getting results
        # cleaned = np.asarray(cleaned)
        pool.close()
        pool.join()
        return cleaned

    return partial(easy_parallize, f)

def plot_fitness(prefix, avg_fit, best_fit):

    plt.clf()
    plt.plot(avg_fit, marker='o', ls='-', color='blue', label='average')
    plt.plot(best_fit, marker='o', color='black', label='best')
    plt.ylabel('fitness (-NRMSE)')
    plt.xlabel('iteration')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(prefix + '_fitness_over_time.png', dpi=300)
    plt.clf()
    plt.close()

def plot_std(prefix, avg_std, avg_best_std):

    plt.clf()
    plt.plot(avg_std, marker='o', ls='-', color='blue', label='average')
    plt.plot(avg_best_std, marker='o', color='black', label='best')
    plt.ylabel('$\sigma$')
    plt.xlabel('iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + '_std_over_time.png', dpi=300)
    plt.clf()
    plt.close()

def evolve_ESN(prefix, lag, num_groups, delta_t, training_time, cut, evo_trials, fitness_trials, max_iter, mutation_perturbation):

    # Evolution setup - record evolution progress
    best_solution = None
    best_solutions = []
    listRuns_averageFitness = []
    listRuns_bestFitness = []
    listRuns_avgstd = []
    listRuns_beststd = []

    # Run evolution
    obj_funct = MackeyGlassObjective.MackeyPrediction(num_groups, trials=fitness_trials, lag=lag, delta_t=delta_t, training_time=training_time, \
        cut=cut)
    for i in xrange(evo_trials):
        print "evo trial: ", i
        esn_evolver = EvolveReservoir.EvolveReservoir(obj_funct, num_groups=num_groups, population_size=20, mutation_perturbation=mutation_perturbation)
        esn_evolver.evolve(max_iter)

        listRuns_averageFitness.append(esn_evolver.timeseries_avgfitness)
        listRuns_bestFitness.append(esn_evolver.timeseries_bestfitness)
        listRuns_avgstd.append(esn_evolver.timeseries_avgstd)
        listRuns_beststd.append(esn_evolver.timeseries_beststd)

        best_solutions.append(copy.deepcopy(esn_evolver.best_member))
        if i == 0:
            best_solution = copy.deepcopy(esn_evolver.best_member)
        else:
            if best_solution['fitness'] < esn_evolver.best_member['fitness']:
                best_solution = copy.deepcopy(esn_evolver.best_member)

    # Plot evolutionary results
    plot_fitness(prefix, np.mean(listRuns_averageFitness, axis=0), np.max(listRuns_bestFitness, axis=0))
    plot_std(prefix, np.mean(listRuns_avgstd, axis=0), np.mean(listRuns_beststd, axis=0))

    # Save best solution   
    save_object(best_solutions, prefix + "_all_best_solutions.pyobj")
    save_object(best_solution, prefix + "_best_solution.pyobj")
    
    # Return best agent fitnesses
    return np.max(listRuns_bestFitness, axis=1)

def multi_lag_evolutionary_performance(prefix, num_groups, lag_range, num_lags, delta_t, training_time, \
    cut, evo_trials, fitness_trials, max_iter, mutation_perturbation):

    # for each lag run an evolution
    listLags = []
    listLags_listRuns_bestfitness = []
    for i, lag in enumerate():
        lag_prefix = prefix + "_" + str(i) + "_"
        listLags_listRuns_bestfitness.append(evolve_ESN(lag_prefix, lag, num_groups, delta_t, training_time, cut, \
            evo_trials, fitness_trials, max_iter, mutation_perturbation))
        listLags.append(lag)

def plot_evolution_over_lags(prefix, listLags, listLags_listRuns_bestfitness):

    # Get quantiles for each lag
    upper_q_list = []
    med_list = []
    lower_q_list = []
    for fitness_list in listLags_listRuns_bestfitness:
        low, med, high = np.percentile(fitness_list, [25,50,75])
        upper_q_list.append(high)
        med_list.append(med)
        lower_q_list.append(low)

    # Plot median performance per delay w/ 95% quantiles (shaded)
    plt.clf()
    plt.plot(listLags, med_list, marker='o', ls='-', color='#CC4F1B', markersize=4)
    plt.ylabel('-NRMSE')
    plt.xlabel('$lag$')
    plt.fill_between(listLags, lower_q_list, upper_q_list, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.4)
    plt.tight_layout()
    plt.savefig(prefix + '_evolutionary_bestfitness_vs_delay.png', dpi=300)
    plt.close()
    plt.clf()

# Pick number of Threads to use when running in parallel.
NUM_CPU = 20
if __name__ == '__main__':

    """
    Evolutionary parameters:
    N - size of network 
    group_fraction - sequence of relative sizes of groups 
    degree_dist_shape - gamma distributions shape for the degree (I disabled this for simplicity)
    degree_dist_scale - gamma distributions scale for the degree (I disabled this for simplicity)
    spectral_radius - a reservoir weight scaling constant
    block_element - the range of values that can be take within a block matrix, usually between 0 and 1
    inhib_ratio - the fraciton of inhibitory connections
    w_in_bias - the input weight bias (a constant multiple of the input signal)
    w_in_fraction - the fraction of random neurons that recieve the input signal
    w_gamma_shape - shape of reservoir weight distribution (gamma distributed) 
    w_gamma_scale - scale of reservoir weight distribution
    leak_rate - scale leack rate for CRTNN
    tc_gamma_shape - shape of gamma distribution used for drawing time constants for each neuron 
    tc_gamma_scale - scale " " "
    random_input - If True input signal is directed to whole network, if False it is only directed to group 0
    """

    #===========================================================================
    # Single lag evolution - NOT PARALLEL
    #===========================================================================

    # # Glass parameters
    # num_groups = 5
    # lag = 100 # Positive lag is predictive, negative lag is recall
    # delta_t = 0.1 # step size
    # training_time = 2000
    # cut = 300 # Cut out initial transient
    # prefix = "test" # Prefix for output files
    # evo_trials = 1 # Number of evolutionary algorithms to run (for averaging)
    # fitness_trials = 1 # Number of times to draw and test reservoirs from a population member
    # max_iter = 10 # Duration of genetic evolution
    # mutation_perturbation = 0.1 # Mutation rate
    # # Set range to determine how the [0,1] reals are converted to parameters values.
    # # parameters can effectively be removed from evolution by setting the min and max
    # # of the range to a desired value.
    # obj_pars = {'num_groups':num_groups, 
    #     'trials':fitness_trials, 
    #     'lag':lag, 
    #     'delta_t':delta_t, 
    #     'training_time':training_time, 
    #     'cut':cut, 
    #     'N_range':(100.0,100.0), 
    #     'group_fraction_range':(1.0,1.0), 
    #     'degree_dist_shape_range':(0.0,0.0), 
    #     'degree_dist_scale_range':(0.0,0.0),
    #     'spectral_radius_range':(0.1,2),
    #     'block_element_range':(0.001, 1.0),
    #     'inhib_ratio_range':(0.0,0.0), 
    #     'W_in_bias_range':(1.0,1.0), 
    #     'W_in_fraction_range':(0.1,0.1), 
    #     'w_gamma_shape_range':(2.0,2.0),
    #     'w_gamma_scale_range':(2.0,2.0), 
    #     'leak_rate_range':(1.0,1.0), 
    #     'tc_gamma_shape_range':(2.0,2.0), 
    #     'tc_gamma_scale_range':(2.0,2.0),
    #     'random_input':True}

    # # Single lag evolution - Perform evolution
    # evolve_ESN(prefix, lag, num_groups, delta_t, training_time, cut, evo_trials, fitness_trials, max_iter, mutation_perturbation)
    # # Run test simulation of best
    # best_solution = load_object(prefix + "macky_best.dat")
    # # Test best genome
    # tester = DiscreteESNTester.ESNTest(best_solution['genome'], prefix=prefix, num_groups=num_groups, lag=lag, delta_t=delta_t, training_time=training_time, cut=cut)
    # print "Testing reservoir performance..."
    # tester.ReservoirPerformanceTest(100)
    # print "Testing reservoir lag performance..."
    # tester.ReservoirLagPerformanceTest((1,1000), 30, 100)
    # print "Testing no-retraining lag performance..."
    # tester.NoRetrainingLagPerformanceTest((1,1000), 30, 100)

    #===========================================================================
    # Parallel single lag evolution
    #===========================================================================
    # num_groups = 1
    # lag = -250
    # delta_t = 0.01
    # training_time = 2000
    # cut = 300
    # prefix = "oneGroup_evoBlock_L-250_gen200"
    # evo_trials = 20
    # fitness_trials = 10
    # max_iter = 200
    # mutation_perturbation = 0.1
    # obj_pars = {'num_groups':num_groups, 
    #     'trials':fitness_trials, 
    #     'lag':lag, 
    #     'delta_t':delta_t, 
    #     'training_time':training_time, 
    #     'cut':cut, 
    #     'N_range':(100.0,100.0), 
    #     'group_fraction_range':(1.0,1.0), 
    #     'degree_dist_shape_range':(0.0,0.0), 
    #     'degree_dist_scale_range':(0.0,0.0),
    #     'spectral_radius_range':(0.1,2),
    #     'block_element_range':(0.001, 1.0),
    #     'inhib_ratio_range':(0.0,0.0), 
    #     'W_in_bias_range':(1.0,1.0), 
    #     'W_in_fraction_range':(0.1,0.1), 
    #     'w_gamma_shape_range':(2.0,2.0),
    #     'w_gamma_scale_range':(2.0,2.0), 
    #     'leak_rate_range':(1.0,1.0), 
    #     'tc_gamma_shape_range':(2.0,2.0), 
    #     'tc_gamma_scale_range':(2.0,2.0),
    #     'random_input':True}

    # def parallel_evolve(ID):

    #     # Re-seed
    #     np.random.seed()
    #     rnd.seed()

    #     obj_funct = MackeyGlassObjective.MackeyPrediction(**obj_pars)
    #     esn_evolver = EvolveReservoir.EvolveReservoir(obj_funct, num_groups=num_groups, population_size=20, mutation_perturbation=mutation_perturbation)
    #     esn_evolver.evolve(max_iter)

    #     return (esn_evolver.timeseries_avgfitness, esn_evolver.timeseries_bestfitness, \
    #         esn_evolver.timeseries_avgstd, esn_evolver.timeseries_beststd, esn_evolver.best_member)

    # parallel_evolve.parallel = parallel_function(parallel_evolve)
    # results = parallel_evolve.parallel(range(evo_trials))

    # # Unwrap results
    # listRuns_averageFitness, listRuns_bestFitness, listRuns_avgstd, listRuns_beststd, best_solutions = zip(*results)
    # best_solution = max(best_solutions, key=lambda member: member['fitness'])

    # # Save best solution
    # save_object(MackeyGlassObjective.MackeyPrediction(**obj_pars), prefix + "_objective.pyobj")
    # save_object(best_solutions, prefix + "_all_best_solution.pyobj")
    # save_object(best_solution, prefix + "_best_solution.pyobj")

    # # Plot evolutionary results
    # plot_fitness(prefix, np.mean(listRuns_averageFitness, axis=0), np.max(listRuns_bestFitness, axis=0))
    # plot_std(prefix, np.mean(listRuns_avgstd, axis=0), np.mean(listRuns_beststd, axis=0))

    # # Run test simulation of best
    # best_solution = load_object(prefix + "_best_solution.pyobj")
    # obj_funct = load_object(prefix + "_objective.pyobj")
    # # Test best genome
    # tester = DiscreteESNTester.ESNTest(best_solution['genome'], obj_funct, prefix)
    # print prefix + " Testing reservoir performance..."
    # tester.ReservoirPerformanceTest(100)
    # print "Testing reservoir lag performance..."
    # tester.ReservoirLagPerformanceTest((1,1000), 30, 25)
    # print "Testing no-retraining lag performance..."
    # tester.NoRetrainingLagPerformanceTest((1,1000), 30, 25)

    #===========================================================================
    # Solution Network analysis
    #===========================================================================
    # prefix = "rndIn_evoBlock_L-250_gen200"
    # best_solution = load_object(prefix + "_best_solution.pyobj")
    # objective = load_object(prefix + "_objective.pyobj")
    # tester = DiscreteESNTester.ESNTest(best_solution['genome'], objective, prefix)
    # tester.Test()
    # tester.print_genome()
    # tester.plot_connectivity_block_matrix()
    # tester.generate_graph()

    # objective = load_object(prefix + "_objective.pyobj")
    # best_solutions = load_object(prefix + "_all_best_solution.pyobj")
    # connectivity_avg = []
    # for i, solution in enumerate(best_solutions):
    #     tester = DiscreteESNTester.ESNTest(solution['genome'], objective, prefix=str(i) + prefix)
    #     tester.print_genome()
    #     tester.plot_connectivity_block_matrix()
    #     connectivity_avg.append(tester.connectivity_block_matrix)

    # avg_matrix = np.mean(connectivity_avg, axis=0)
    # tester.connectivity_block_matrix = avg_matrix
    # tester.prefix = "avg_block_matrix"
    # tester.plot_connectivity_block_matrix()

    #===========================================================================
    # Network Flow Analysis
    #===========================================================================
    # prefix = "rndIn_evoBlock_L-250_gen200"
    # objective = load_object(prefix + "_objective.pyobj")
    # best_solutions = load_object(prefix + "_all_best_solution.pyobj")
    # connectivity_avg = []
    # network_flow = []
    # for i, solution in enumerate(best_solutions):
    #     tester = DiscreteESNTester.ESNTest(solution['genome'], objective, prefix=str(i) + prefix)
    #     network_flow.append(abs(np.sum(np.triu(tester.connectivity_block_matrix)) - np.sum(np.tril(tester.connectivity_block_matrix))))
    # print "Data: ", stats.bayes_mvs(network_flow, alpha=0.95)[0]

    # rnd_net_flow = []
    # for i in xrange(len(best_solutions)):
    #     rnd_block = np.random.uniform(0.001, 1.0, size=(5,5))
    #     rnd_net_flow.append(abs(np.sum(np.triu(rnd_block)) - np.sum(np.tril(rnd_block)) ))
    # print "NULL: ", stats.bayes_mvs(rnd_net_flow, alpha=0.95)[0]


    #===========================================================================
    # Parallel multi lag evolution
    #===========================================================================
    # lag_range = (10, 1000)
    # num_lags = 20
    # delta_t = 0.1
    # training_time = 2000
    # cut = 300
    # prefix = "lag10-1000_mutate0.1"
    # evo_trials = 1
    # fitness_trials = 10
    # max_iter = 100
    # mutation_perturbation = 0.1

    # def parallel_lag_evolution(lag):

    #     # Re-seed
    #     np.random.seed()
    #     rnd.seed()
        
    #     lag_prefix = prefix + "_" + str(lag) + "_"
    #     best = evolve_ESN(lag_prefix, lag, num_groups, delta_t, training_time, cut, \
    #         evo_trials, fitness_trials, max_iter, mutation_perturbation)

    #     return (lag, best)

    # parallel_lag_evolution.parallel = parallel_function(parallel_lag_evolution)
    # results = parallel_lag_evolution.parallel(np.linspace(lag_range[0], lag_range[1], num_lags))

    # # Unwrap results
    # lags, listLags_listRuns_bestfitness = zip(*results)

    # # Plot results
    # plot_evolution_over_lags(prefix, lags, listLags_listRuns_bestfitness)

    #===========================================================================
    # Base-line performance
    #===========================================================================

    # # Params
    # lag = 2000
    # delta_t = 0.1
    # beta = 0.2
    # gamma = 0.1
    # n = 10.0
    # tau = 17.0
    # training_time = 20000

    # def generate_test_glass():
    #     # Construct MackeyGlass
    #     ic = np.random.uniform(0.0,1.0, size=int(tau / delta_t) + 1)
    #     macky = DiscreteMackeyGlass.MackeyGlassSystem(ic, delta_t, beta, gamma, n, tau)

    #     # Generate time-series (squash between -1 and 1)
    #     time_series = np.tanh(np.array(macky.generate_series(int(training_time / delta_t))) - 1.)
    #     time_series = time_series.reshape((len(time_series), 1, 1))

    #     # Generate target time-series (and adjust for lag)
    #     target_time_series = time_series[int(math.ceil(lag / delta_t) + 0.4):]
    #     time_series = time_series[:-int(math.ceil(lag / delta_t) + 0.4)]

    #     return time_series, target_time_series

    # def NormalizedRootMeanSquaredError(residuals, ymax, ymin):
        
    #     return np.sqrt(np.mean(np.power(residuals, 2))) / (ymax - ymin)

    # training_ts, training_tar_ts = generate_test_glass()
    # mean_model = np.mean(training_ts) * np.ones(training_ts.shape)
    # valid_ts, valid_tar_ts = generate_test_glass()
    # test_residuals = np.abs(mean_model - valid_tar_ts)
    # nrmse = NormalizedRootMeanSquaredError(test_residuals, np.max(valid_tar_ts), np.min(valid_tar_ts))
    # print -nrmse

    #===========================================================================
    # Best-member tester!
    #===========================================================================

    # # Run test simulation of best
    # best_solution = load_object("lag100_mutate0.1macky_best.dat")
    # # Test best genome
    # objective = load_object(prefix + "_objective.pyobj")
    # tester = DiscreteESNTester.ESNTest(best_solution['genome'], objective, prefix)
    # tester.print_genome()
    # print "Testing reservoir performance..."
    # # tester.ReservoirPerformanceTest(100)
    # print "Testing reservoir lag performance..."
    # tester.ReservoirLagPerformanceTest((1,1000), 5, 51)
    # print "Testing no-retraining lag performance..."
    # # tester.NoRetrainingLagPerformanceTest((1,1000), 5, 51)


    #===========================================================================
    # Time series size of best-member!
    #===========================================================================
    
    # # Load from directory
    # best_files = [ file for file in os.listdir(".") if ("lag10-1000" in file) and ("macky_best" in file)]

    # # Get size of each
    # lags = []
    # sizes = []
    # for best_file in best_files: #"lag10-1000_mutate0.1_843.684210526_macky_best.dat"
    #     best_solution = load_object(best_file)
    #     objective = load_object(prefix + "_objective.pyobj")
    #     tester = DiscreteESNTester.ESNTest(best_solution['genome'], objective, prefix)
    #     sizes.append(tester.N)
    #     lags.append(float(best_file.replace("lag10-1000_mutate0.1_","").replace("_macky_best.dat","")))

    # temp = zip(lags, sizes)
    # temp.sort()
    # lags, sizes = zip(*temp)

    # plt.clf()
    # plt.plot(lags, sizes, ls='-', marker='o', color='blue')
    # plt.xlabel("$lag$")
    # plt.ylabel("N")
    # plt.tight_layout()
    # plt.savefig("lag10-1000_size_of_best.png", dpi=300)
    # plt.close()
    # plt.clf()

    #===========================================================================
    # Dry Runs of different networks structures
    #===========================================================================
    # 5 Weak groups - Median Error:  -0.0390443525991
    # 5 strong groups - Median Error:  -0.0909486001314
    # Rnd Median Error:  -0.0390228289657
    # num_groups = 5
    # lag = -100
    # delta_t = 0.1
    # training_time = 2000
    # cut = 300
    # prefix = "rnd"
    # evo_trials = 20
    # fitness_trials = 2 #10
    # max_iter = 200
    # mutation_perturbation = 0.1
    # obj_pars = {'num_groups':num_groups, 
    #     'trials':fitness_trials, 
    #     'lag':lag, 
    #     'delta_t':delta_t, 
    #     'training_time':training_time, 
    #     'cut':cut, 
    #     'N_range':(100.0,100.0), 
    #     'group_fraction_range':(1.0,1.0), 
    #     'degree_dist_shape_range':(0.0,0.0), 
    #     'degree_dist_scale_range':(0.0,0.0),
    #     'spectral_radius_range':(0.1,2),
    #     'block_element_range':(0.001, 1.0),
    #     'inhib_ratio_range':(0.0,0.0), 
    #     'W_in_bias_range':(1.0,1.0), 
    #     'W_in_fraction_range':(0.1,0.1), 
    #     'w_gamma_shape_range':(2.0,2.0),
    #     'w_gamma_scale_range':(2.0,2.0), 
    #     'leak_rate_range':(1.0,1.0), 
    #     'tc_gamma_shape_range':(2.0,2.0), 
    #     'tc_gamma_scale_range':(2.0,2.0),
    #     'random_input':False,
    #     'unique_group_weights':False}
    # obj_funct = MackeyGlassObjective.MackeyPrediction(**obj_pars)
    # blocks = [[0.5, 0.5, 0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5, 0.5, 0.5]]
    # genome = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 
    #     0.0, 0.0, 0.0, 0.0, 0.0] + [ block for i in blocks for block in i ] + [0.0, 0.0, 0.0, 0.0]
    # tester = DiscreteESNTester.ESNTest(genome, obj_funct, prefix)
    # print "Testing reservoir performance..."
    # tester.ReservoirPerformanceTest(100)
    # tester.Test()
    # # tester.print_genome()
    # tester.plot_connectivity_block_matrix()
    # tester.generate_graph()