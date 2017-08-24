import numpy as np 
import bigred2_mkPCPcommand_files

#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]#np.linspace(0.01, 2.0, 6)#[10,20,30,40,50,60]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
experimental_parameters = {
    'num_threads': 320,#512,#736
    'q1_list': np.linspace(0.0, 0.5, 18),#np.linspace(0.01, 0.5, 15),
    'q2_list': [ 80 ],#np.linspace(0.1, 5.0, 10),
    'qsub_nodes': 10,#16,23
    'qsub_ppn': 32,
    'qsub_time': '00:45:00',
    'qsub_cpu_type': 'cpu',
    'worker_file': 'bigred2_nbitworker.py',
    'command_prefix': 'runacrross80',
    'q1': 'mu',
    'q2': 'distraction_duration',
    'num_reservoir_samplings': 17,
    'num_validation_trials': 256,
    'nbit_task_parameters': {
    'loop_unique_input': True,
    'distractor_value': 0,
    'sequence_dimension': 4,
    'start_time': 0,
    'sequence_length': 4,#4
    'distraction_duration': 30,
    'input_fraction': 0.3,
    'input_gain': 2.0,
    'num_trials': 256,
    'neuron_type': 'sigmoid',
    'neuron_pars': {'c':1., 'e': 10},
    'output_neuron_type': 'heaviside',
    'output_neuron_pars': {'threshold': 0.5},
    'N': 1000,
    'mu': 0.1,
    'k': 7, 
    'maxk': 7,
    'homok': None,
    'com_size': None, 
    'minc':10, 
    'maxc':10, 
    'deg_exp':1.0, 
    'temp_dir_ID':0,
    'full_path': '/N/u/njrodrig/BigRed2/topology_of_function/',#'/nobackup/njrodrig/topology_of_function/',
    'reservoir_weight_scale': 1.0,
    'input_weight_bounds': (1.0,1.0),#(1.0, 1.0),
    'lower_reservoir_bound': -0.1,
    'upper_reservoir_bound': 1.0
    }
}

bigred2_mkPCPcommand_files.run_qsub(experimental_parameters)