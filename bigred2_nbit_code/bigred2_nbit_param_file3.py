import numpy as np 
import bigred2_mkPCPcommand_files

experimental_parameters = {
    'num_threads': 64,#736
    'q1_list': np.linspace(0.0, 0.5, 10),
    'q2_list': [5, 10], #17
    'qsub_nodes': 2,#16,23
    'qsub_ppn': 32,
    'qsub_time': '00:50:00',
    'qsub_cpu_type': 'debug_cpu',
    'worker_file': 'bigred2_nbitworker.py',
    'command_prefix': 'test',
    'q1': 'mu',
    'q2': 'distraction_duration',
    'num_reservoir_samplings': 3,
    'num_validation_trials': 256,
    'fixed_point_analysis': False,
    'nbit_task_parameters': {
    'loop_unique_input': True,
    'recall_task': True,
    'distractor_value': 0,
    'cue_value':0,
    'sequence_dimension': 4,
    'start_time': 0,
    'sequence_length': 5,
    'distraction_duration': 1000,
    'input_fraction': 0.3,
    'input_gain': 2.0,
    'num_trials': 200,
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