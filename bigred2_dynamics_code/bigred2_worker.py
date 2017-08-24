import sys
import random
import numpy as np
import utilities
import graphgen
import echostatenetwork
import fixed_point_estimator

def worker(experimental_parameters):

    random.seed()
    np.random.seed()

    results = []
    for parameter_set in experimental_parameters:
        reservoir = graphgen.uniform_weighted_directed_lfr_graph_asarray(N=parameter_set['N'], 
            mu=parameter_set['mu'], 
            k=parameter_set['k'], maxk=parameter_set['maxk'], minc=parameter_set['minc'], 
            maxc=parameter_set['maxc'], deg_exp=parameter_set['deg_exp'], 
            temp_dir_ID=parameter_set['temp_dir_ID'], full_path=parameter_set['full_path'], 
            weight_bounds=parameter_set['reservoir_weight_scale'] * np.array(parameter_set['reservoir_weight_bounds']))
        esn = echostatenetwork.DESN(reservoir, neuron_type=parameter_set['neuron_type'], 
            neuron_pars=parameter_set['neuron_pars'], init_state=parameter_set['ic'])
        
        results.append((parameter_set[parameter_set['q1']], 
            parameter_set[parameter_set['q2']], 
            fixed_point_estimator.fixed_point_estimate(esn, parameter_set['num_trials'], 
                parameter_set['tmax'], parameter_set['distance_thresh'], parameter_set['epsilon'], 
                parameter_set['convergence_delay']) ))

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