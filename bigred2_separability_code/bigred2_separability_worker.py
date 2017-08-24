import sys
import random
import numpy as np
import utilities
import graphgen
import echostatenetwork
import esn_separability
import memory_capacity_objective

def worker(experimental_parameters):

    random.seed()
    np.random.seed()

    parameter_set_results = []
    for parameter_set in experimental_parameters:
        task_master = esn_separability.SeparabilityAnalyzer(**parameter_set)

        # Evaluate separability
        separability_results = task_master.evaluate_trials_over_all_distances()
        # Evaluate performance
        parameter_set['premad_esn'] = {'esn': task_master.esn,
            'reservoir': task_master.reservoir,
            'nx_reservoir': task_master.nx_reservoir,
            'input_weights': task_master.input_weights
        }
        performance_task = memory_capacity_objective.memory_capacity_objective(**parameter_set)
        performance_task.train()
        performance_results = performance_task.validate()

        parameter_set_results.append((parameter_set[parameter_set['q1']],
            parameter_set[parameter_set['q2']],
            separability_results,
            task_master.euclidean_range, performance_results))

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