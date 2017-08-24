import bigred2_mc_process
import numpy as np

experimental_parameters = {
    'q1_list': np.linspace(0.0, 0.5, 20),
    'q2_list': np.linspace(0.1, 0.6, 20),
    'q1': 'mu',
    'q2': 'input_fraction',
    'training_length': 1000,
    'validation_length': 500,
    'binary':True,
    'cut': 100,
    'max_delay': 25,
    'shift': 1,
    'input_range': (0.0, 1.0),
    'input_fraction': 0.6,
    'input_gain': 1.0,
    'input_weight_bounds': (-0.2, 1.0),
    'target_spectral_radius': None,
    'neuron_type': 'sigmoid',
    'neuron_pars': {'c':1, 'e':10},
    'output_neuron_type': 'heaviside',
    'output_neuron_pars': {'threshold': 0.5},
    'num_reservoir_samplings': 64,
    'N': 500,
    'mu': 0.5,
    'k': 6,
    'maxk': 6,
    'minc':10,
    'maxc':10,
    'deg_exp':1.0,
    'lower_reservoir_bound': -0.2,
    'upper_reservoir_bound': 1.0,
    'reservoir_weight_scale': 1.132,
    'temp_dir_ID':0, 
    'full_path': '/N/u/njrodrig/BigRed2/topology_of_function/',#'/nobackup/njrodrig/topology_of_function/',
    'worker_file': 'bigred2_mc_worker.py',
    'num_threads': 1024,
    'command_prefix': 'MC_N500_c1_e10_train1k_ws1.132_gain1.0_com10_mu_v_rsig_version2',
    'qsub_nodes': 32,
    'qsub_ppn': 32,
    'qsub_time': '06:00:00',
    'qsub_cpu_type': 'cpu'
}

bigred2_mc_process.run_qsub(experimental_parameters)