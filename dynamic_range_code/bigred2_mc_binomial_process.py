import numpy as np
import copy
import utilities
import subprocess
import os

def generate_parallel_input_sequence(list_q1, list_q2, experimental_parameters):
    
    parallel_input_sequence = []
    for i, q1 in enumerate(list_q1):
        for j, q2 in enumerate(list_q2):
            for k, sampleID in enumerate(xrange(experimental_parameters['num_reservoir_samplings'])):
                experimental_parameters[experimental_parameters['q1']] = q1
                experimental_parameters[experimental_parameters['q2']] = q2
                experimental_parameters['temp_dir_ID'] = experimental_parameters['command_prefix'] + \
                    str(i * len(list_q2) * experimental_parameters['num_reservoir_samplings'] + \
                    j * experimental_parameters['num_reservoir_samplings'] + k)
                parallel_input_sequence.append(copy.deepcopy(experimental_parameters))

    return parallel_input_sequence

def write_command_file(worker_file, command_prefix, num_chunks, full_path):

    execution_paths = [full_path + worker_file]*num_chunks
    object_paths = [full_path + command_prefix + "_" + str(i) + ".pyobj" for i in xrange(num_chunks)]
    chunk_numbers = [ str(i) for i in xrange(num_chunks) ]

    utilities.writecol(full_path + command_prefix + "_command_file.txt", " ", "", ["python"]*len(execution_paths), 
        execution_paths, object_paths, chunk_numbers)

    return object_paths

def write_parameter_files(parameter_file_names, parallel_input_chunks):

    for i in xrange(len(parallel_input_chunks)):
        utilities.save_object(parallel_input_chunks[i], parameter_file_names[i])

def write_output_filenamelist(command_prefix, num_chunks, full_path):

    output_paths = [ full_path + command_prefix + "_output" + str(i) + ".pyobj" \
        for i in xrange(num_chunks) ]

    utilities.save_object(output_paths, full_path + command_prefix + "_outputfilelist.pyobj")

def write_qsub_script(parameters):
    
    file_text = "#!/bin/bash\n"
    file_text += "#PBS -l nodes=" + str(parameters['qsub_nodes']) + ":ppn=" + str(parameters['qsub_ppn']) + "\n"
    file_text += "#PBS -l walltime=" + parameters['qsub_time'] + "\n"
    file_text += "#PBS -N " + parameters['command_prefix'] + "\n"
    file_text += "#PBS -q " + parameters['qsub_cpu_type'] + "\n"
    file_text += "#PBS -V\n"
    file_text += "#PBS -M njrodrig@umail.iu.edu\n"
    file_text += "#PBS -m abe\n"
    file_text += "#PBS -j oe\n"
    file_text += "#PBS -k o\n"

    file_text += "module load pcp/2008\n"
    file_text += "cd /N/u/njrodrig/BigRed2/topology_of_function\n"
    file_text += "aprun -n " + str(parameters['num_threads']) + " pcp " + parameters['full_path'] + \
        parameters['command_prefix'] + "_command_file.txt\n"
    file_text += "aprun -n 1 python bigred2_mc_binomial_consolidation.py " + \
        parameters['command_prefix'] + " " + parameters['full_path']

    script_name = "bigred2_" + parameters['command_prefix'] + ".script"
    file = open(script_name, 'w')
    file.write(file_text)
    file.close()

    return script_name

def run_qsub(parameters):

    setup_parameter_files(parameters)
    script_name = write_qsub_script(parameters)

    subprocess.Popen(['qsub', script_name], preexec_fn=os.setpgrp)

def setup_parameter_files(parameters):

    # 1 is subtracted because pcp only runes N-1 processes in parallel.
    max_proc = min(parameters['num_threads']-1, len(parameters['q1_list']) * len(parameters['q2_list']) * 
        parameters['num_reservoir_samplings'])

    parallel_input_sequence = generate_parallel_input_sequence(parameters['q1_list'], parameters['q2_list'], parameters)
    parallel_input_chunks = utilities.split_list(parallel_input_sequence, max_proc)
    parameter_file_names = write_command_file(parameters['worker_file'], parameters['command_prefix'], 
        len(parallel_input_chunks), parameters['full_path'])

    write_parameter_files(parameter_file_names, parallel_input_chunks)
    write_output_filenamelist(parameters['command_prefix'], len(parameter_file_names), 
        parameters['full_path'])
    utilities.save_object({'parameters': parameters, 'q1_list': parameters['q1_list'], 'q2_list': parameters['q2_list']}, 
        parameters['full_path'] + parameters['command_prefix'] + "_paramfile.pyobj")