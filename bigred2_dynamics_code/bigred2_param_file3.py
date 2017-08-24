import bigred2_process
import numpy as np

experimental_parameters = {
    'q1_list': np.linspace(0.0, 0.5, 20),
    'q2_list': [1.132],
    'q1': 'mu',
    'q2': 'reservoir_weight_scale',
    'num_reservoir_samplings': 12,
    'num_trials': 800,
    'neuron_type': 'sigmoid',
    'neuron_pars': {'c':1, 'e':10},
    'N': 500,
    'mu': 0.1,
    'k': 6,
    'maxk': 6,
    'minc':10,
    'maxc':10,
    'deg_exp':1.0,
    'ic': (0.0, 0.8),
    'tmax': 1000,
    'distance_thresh': 0.01,
    'epsilon': 0.01,
    'convergence_delay': 10,
    'temp_dir_ID': 0,
    'full_path': '/N/u/njrodrig/BigRed2/topology_of_function/',#'/nobackup/njrodrig/topology_of_function/',
    'reservoir_weight_scale': 1.132,
    'reservoir_weight_bounds': (-0.2, 1.0),
    'worker_file': 'bigred2_worker.py',
    'num_threads': 256,#736
    'command_prefix': 'fp_ic0-0.8_tmax1k_N500_c1_e10_train1k_ws1.132_mu',
    'qsub_nodes': 8,#16,23
    'qsub_ppn': 32,
    'qsub_time': '01:00:00',
    'qsub_cpu_type': 'cpu'
}

bigred2_process.run_qsub(experimental_parameters)