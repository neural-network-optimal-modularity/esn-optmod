import bigred2_mc_process
import numpy as np

experimental_parameters = {
    'q1_list': np.linspace(-15, 0.0, 15),
    'q2_list': np.linspace(0.01, 15.0, 15),
    'q1': 'lower_reservoir_bound',
    'q2': 'upper_reservoir_bound',
    'training_length': 500,
    'validation_length': 500,
    'cut': 100,
    'max_delay': 50,
    'shift': 2,
    'input_range': (0.0, 1.0),
    'input_fraction': 0.3,
    'input_gain': 0.5,
    'input_weight_bounds': (0.0, 1.0),
    'neuron_type': 'sigmoid',
    'neuron_pars': {'c':1, 'e':10},
    'output_neuron_type': 'identity',
    'output_neuron_pars': {},
    'num_reservoir_samplings': 1,
    'N': 500,
    'mu': 0.2,
    'k': 6,
    'maxk': 6,
    'minc':10,
    'maxc':10,
    'deg_exp':1.0,
    'lower_reservoir_bound': -0.1,
    'upper_reservoir_bound': 1.0,
    'reservoir_weight_scale': 1.0,
    'temp_dir_ID':0, 
    'full_path': '/N/u/njrodrig/BigRed2/topology_of_function/',#'/nobackup/njrodrig/topology_of_function/',
    'worker_file': 'bigred2_mc_worker.py',
    'num_threads': 256,
    'command_prefix': 'MC_N500_c1_e10_in0_1_train500_lrb-urb',
    'qsub_nodes': 8,
    'qsub_ppn': 32,
    'qsub_time': '00:45:00',
    'qsub_cpu_type': 'cpu'
}

bigred2_mc_process.run_qsub(experimental_parameters)