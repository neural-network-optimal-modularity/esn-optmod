import sys
import pickle
import task_objectives
import random
import numpy as np

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

def worker(experimental_parameters):

    random.seed()
    np.random.seed()

    results = []
    for parameter_set in experimental_parameters:
        if parameter_set['task_type'] == 'binary_memory_objective':
            task_master = task_objectives.binary_memory_objective(**parameter_set['task_parameters'])
        elif parameter_set['task_type'] == 'poly_objective':
            task_master = task_objectives.poly_objective(**parameter_set['task_parameters'])
        elif parameter_set['task_type'] == 'find_largest_objective':
            task_master = task_objectives.find_largest_objective(**parameter_set['task_parameters'])
        elif parameter_set['task_type'] == 'null_objective':
            task_master = task_objectives.null_objective(**parameter_set['task_parameters'])

        task_master.TrainModel()
        results.append((parameter_set['task_parameters'][parameter_set['q1']], 
            parameter_set['task_parameters'][parameter_set['q2']],
            task_master.ValidateModel(parameter_set['num_validation_trials'])))

    return results

def main(argv):

    experiment_parameter_file = str(argv[0])
    chunkID = str(argv[1])

    experimental_parameters = load_object(experiment_parameter_file)
    results = worker(experimental_parameters)

    save_object(results, experimental_parameters[0]['task_parameters']['full_path'] +\
        experimental_parameters[0]['command_prefix'] + "_output" + chunkID + ".pyobj")

if __name__ == '__main__':
    main(sys.argv[1:])