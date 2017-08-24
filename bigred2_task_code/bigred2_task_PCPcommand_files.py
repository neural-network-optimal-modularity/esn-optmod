import numpy as np
import copy
import pickle

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

def chunks_list(a, n):
    """
    returns list instead of a generator
    """

    k, m = len(a) / n, len(a) % n
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]

def writecol(filename,delimiter,firstline,*args):
    """
    writecol(filename,delimiter,firstline,*args)

    Writes to file a list of columns provided as arguments to the function.
    If input is provided for firstline that is not "", then that string
    is made the first line of the file. Columns must be of same length 
    
    filename
        file to be read
    delimiter
        character or characters used as a delimiter between columns
    firstline
        header for file, if set to '', then none is written
    *args
        lists of columns to be written to the text file
    
    """
    
    col = [arg for arg in args]
    
    # Make sure columns are of the same length
    for x in col:
        assert(len(x) == len(col[0]))
    
    lines = []
    if firstline != "":
        lines.append(firstline + '\n')
    
    col_num = range(0,len(col))
    end = col_num[-1]
    for i in range(0,len(col[0])):
        line = ''
        for j in col_num:
            if j == end:
                line += str(col[j][i])
            else:
                line += str(col[j][i]) + delimiter
        line += '\n' 
        lines.append(line)
    
    outfile = open(filename,'w')
    outfile.writelines(lines)
    outfile.close()

def generate_parallel_input_sequence(list_q1, list_q2, experimental_parameters):
    
    parallel_input_sequence = []
    for i, q1 in enumerate(list_q1):
        for j, q2 in enumerate(list_q2):
            for k, sampleID in enumerate(xrange(experimental_parameters['num_reservoir_samplings'])):
                experimental_parameters['task_parameters'][experimental_parameters['q1']] = q1 
                experimental_parameters['task_parameters'][experimental_parameters['q2']] = q2 
                experimental_parameters['task_parameters']['temp_dir_ID'] = experimental_parameters['command_prefix'] + \
                    str(i * len(list_q2) * experimental_parameters['num_reservoir_samplings'] + \
                    j * experimental_parameters['num_reservoir_samplings'] + k)
                parallel_input_sequence.append(copy.deepcopy(experimental_parameters))

    return parallel_input_sequence

def write_command_file(worker_file, command_prefix, num_chunks, full_path):

    execution_paths = [full_path + worker_file]*num_chunks
    object_paths = [full_path + command_prefix + "_" + str(i) + ".pyobj" for i in xrange(num_chunks)]
    chunk_numbers = [ str(i) for i in xrange(num_chunks) ]

    writecol(full_path + command_prefix + "_command_file.txt", " ", "", ["python"]*len(execution_paths), 
        execution_paths, object_paths, chunk_numbers)

    return object_paths

def write_parameter_files(parameter_file_names, parallel_input_chunks):

    for i in xrange(len(parallel_input_chunks)):
        save_object(parallel_input_chunks[i], parameter_file_names[i])

def write_output_filenamelist(command_prefix, num_chunks, full_path):

    output_paths = [ full_path + command_prefix + "_output" + str(i) + ".pyobj" \
        for i in xrange(num_chunks) ]

    save_object(output_paths, full_path + command_prefix + "_outputfilelist.pyobj")

if __name__ == '__main__':
    """
    This script writes a series of files out for use on Big Red II.
    The files are used in PCP module in Big Red II to run serial programs
    in parallel across multiple nodes.
    A command list file is written which the aprun + PCP module read and use
    A series of parameter files are written which the worker programs run
    It also writes a pyobj file that contains a list of the output file names that 
    the workers will write their output too.
    It also saves the experimental parameters to file
    """
    command_prefix = 'binary_memory_objective_mu-rsig2'
    worker_file = 'bigred2_taskworker.py'
    list_q1 = np.linspace(0.01, 0.5, 16)
    list_q2 = np.linspace(0.05, 0.5, 12)#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]#np.linspace(0.01, 2.0, 6)#[10,20,30,40,50,60]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]#np.linspace(0.05, ##
    experimental_parameters = {
        'task_type': 'binary_memory_objective',
        'command_prefix': command_prefix,
        'q1': 'mu',
        'q2': 'input_fraction',
        'num_reservoir_samplings': 2,
        'num_validation_trials': 100,
        'task_parameters': {
            'sequence_dimension' : 3,
            'sequence_length': 3,
            'normalized_input': True,
            'distraction_duration': 10,
            'distractor_value': 0,
            'output_neuron_type': 'heaviside',
            'output_neuron_pars': {'threshold': 0.5},
        'input_fraction': 0.2,
        'input_gain': 3.0,
        'num_trials': 200,
        'neuron_type': 'sigmoid',
        'neuron_pars': {'c':1, 'e':10},
        'N': 500, 
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
        'reservoir_weight_bounds': (-0.1, 1.0),
        'input_weight_bounds': (0.0, 1.0)}
    }

    threads_avail = 192
    # 1 is subtracted because pcp only runes N-1 processes in parallel.
    max_proc = min(threads_avail-1, len(list_q1) * len(list_q2) * experimental_parameters['num_reservoir_samplings'])
    parallel_input_sequence = generate_parallel_input_sequence(list_q1, list_q2, experimental_parameters)
    parallel_input_chunks = chunks_list(parallel_input_sequence, max_proc)
    parameter_file_names = write_command_file(worker_file, command_prefix, 
        len(parallel_input_chunks), experimental_parameters['task_parameters']['full_path'])
    write_parameter_files(parameter_file_names, parallel_input_chunks)
    write_output_filenamelist(command_prefix, len(parameter_file_names), 
        experimental_parameters['task_parameters']['full_path'])
    save_object({'experimental_parameters': experimental_parameters, 'list_q1': list_q1, 'list_q2': list_q2}, 
        experimental_parameters['task_parameters']['full_path'] + command_prefix + "_paramfile.pyobj")