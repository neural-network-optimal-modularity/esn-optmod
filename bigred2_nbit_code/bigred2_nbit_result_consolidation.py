import utilities
import sys
import os

def consolidate_data(parameters, output_file_list):

    model_results = []
    for file in output_file_list:
        model_results += utilities.load_object(file)

    num_q1 = len(parameters['q1_list'])
    num_q2 = len(parameters['q2_list'])
    num_trial = parameters['parameters']['num_reservoir_samplings']
    listQ1_listQ2_listTrial_listResults = [ [ [ model_results[i*num_q2*num_trial + j*num_trial + k] 
        for k in xrange(num_trial) ]
        for j in xrange(num_q2) ]
        for i in xrange(num_q1) ]

    return listQ1_listQ2_listTrial_listResults

def cleanup(command_prefix, full_path, output_file_list):

    os.remove(full_path + command_prefix + "_paramfile.pyobj")
    os.remove(full_path + command_prefix + "_command_file.txt")
    os.remove(full_path + command_prefix + "_outputfilelist.pyobj")

    for i, file in enumerate(output_file_list):
        os.remove(file)
        os.remove(full_path + command_prefix + "_" + str(i) + ".pyobj")

def main(argv):

    if len(argv) == 0:
        print """
        Call as: bigred2_result_consolidation command_prefix path
        Where path is the full working directory where the files are locations
        """

    command_prefix = str(argv[0])
    full_path = str(argv[1])

    parameters = utilities.load_object(full_path + command_prefix + "_paramfile.pyobj")
    output_file_list = utilities.load_object(full_path + command_prefix + "_outputfilelist.pyobj")

    results = consolidate_data(parameters, output_file_list)
    utilities.save_object({'parameters': parameters, 'results': results}, full_path + command_prefix + "_final_results.pyobj")

    cleanup(command_prefix, full_path, output_file_list)

if __name__ == '__main__':
    main(sys.argv[1:])