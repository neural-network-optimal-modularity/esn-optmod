import sys
import Nbit_memory_objective
import random
import numpy as np
import utilities

def worker(experimental_parameters):

    random.seed()
    np.random.seed()

    results = []
    for parameter_set in experimental_parameters:
        nbit_task_master = Nbit_memory_objective.Nbit_memory_objective(**parameter_set['nbit_task_parameters'])

        if parameter_set['fixed_point_analysis']:
            results.append((parameter_set['nbit_task_parameters'][parameter_set['q1']], 
                parameter_set['nbit_task_parameters'][parameter_set['q2']],
                nbit_task_master.fixed_point_analysis()))

        else:
            nbit_task_master.TrainModel()
            results.append((parameter_set['nbit_task_parameters'][parameter_set['q1']], 
                parameter_set['nbit_task_parameters'][parameter_set['q2']],
                nbit_task_master.ValidateModel(parameter_set['num_validation_trials'])))

    return results

def main(argv):

    experiment_parameter_file = str(argv[0])
    chunkID = str(argv[1])

    experimental_parameters = utilities.load_object(experiment_parameter_file)
    results = worker(experimental_parameters)

    utilities.save_object(results, experimental_parameters[0]['nbit_task_parameters']['full_path'] +\
        experimental_parameters[0]['command_prefix'] + "_output" + chunkID + ".pyobj")

if __name__ == '__main__':
    main(sys.argv[1:])