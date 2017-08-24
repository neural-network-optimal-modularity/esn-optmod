import sys
import random
import numpy as np
import utilities
import graphgen
import echostatenetwork
import memory_capacity_objective

def worker(experimental_parameters):

    random.seed()
    np.random.seed()

    results = []
    for parameter_set in experimental_parameters:
        task_master = memory_capacity_objective.memory_capacity_objective(**parameter_set)
        task_master.train()
        results.append((parameter_set[parameter_set['q1']], 
            parameter_set[parameter_set['q2']],
            task_master.validate()))

    return results

def main(argv):

    experiment_parameter_file = str(argv[0])
    chunkID = str(argv[1])

    experimental_parameters = utilities.load_object(experiment_parameter_file)
    results = worker(experimental_parameters)
    utilities.save_object(results, experimental_parameters[0]['full_path'] +\
        experimental_parameters[0]['command_prefix'] + "_output" + chunkID + ".pyobj")

if __name__ == '__main__':
    main(sys.argv[1:])