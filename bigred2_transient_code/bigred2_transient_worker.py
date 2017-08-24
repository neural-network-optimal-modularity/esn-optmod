import sys
import random
import numpy as np
import utilities
import graphgen
import echostatenetwork
import esn_transients

def worker(experimental_parameters):

    random.seed()
    np.random.seed()

    parameter_set_results = []
    for parameter_set in experimental_parameters:
        task_master = esn_transients.TransientAnalyzer(**parameter_set)
        parameter_set_results.append((parameter_set[parameter_set['q1']], 
            parameter_set[parameter_set['q2']], task_master.evaluate_reservoir_transient_length_trials()))

    return parameter_set_results

def main(argv):

    experiment_parameter_file = str(argv[0])
    chunkID = str(argv[1])

    experimental_parameters = utilities.load_object(experiment_parameter_file)
    results = worker(experimental_parameters)
    utilities.save_object(results, experimental_parameters[0]['full_path'] +\
        experimental_parameters[0]['command_prefix'] + "_output" + chunkID + ".pyobj")

if __name__ == '__main__':
    main(sys.argv[1:])