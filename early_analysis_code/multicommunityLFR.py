import networkx as nx
import os
import sys
import shutil
import numpy as np

def readcol(filename,format='s',delimiter=None,col=None,skiplines=[]):
    """
    readcol(filename,format='s',delimiter=None,col=None,skiplines=[])

    Reads columns from a file.
    
    filename 
        filename of input file
    format 
        string of format characters (auto formats output)
        s - string
        i - int
        f - float
        Letters must be chosen in the format of a string for each
        column to be output. The default setting for format is 's'.
    delimiter 
        char used to separate columns in a line of text. If None is
        chosen then the full line of a file will be read in.
    col 
        can optionally list the columns you want to return (list of ints).
        By default col=None which means that all columns will be read if
        a delimiter is chosen, or only one if no delimiter is chosen.
    skiplines 
        list of lines to skip in reading.
    
    Returns lists. When multiple columns are chosen the function returns
    each one in the order provided in the col argument. If no col argument was
    given, yet all the columns were read, then a single list containing all the 
    columns is returned.
    
        example: firstcol, secondcol, thrdcol = readcol(file,'sif',',',[1,2,3])
    or
        example cols = readcol(file,'s',',')
    
    """

    # Reject bad input
    if  (delimiter == None) and (col != None): 
        if (len(format) == len(col)):
            print "Must have delimiter for multiple columns."
            sys.exit()

    # Open file and read lines into variables.
    if os.path.exists(filename):
        infile = open(filename,'r')
    else:
        print "The file named: " + filename + " does not exist."
        sys.exit()
    
    # Default read all columns as a single format
    # This requires a delimiter to be defined, col set to None
    if (delimiter != None) and (col == None) and (len(format) == 1):
        data = []
        for i,line in enumerate(infile.readlines()):
            if (i+1 in skiplines) == False: 
                fields = line.strip().split(delimiter)
                fields = [field for field in fields if field != ""]
                row = []
                for str in fields:
                    row.append(str)
                data.append(row)
        # Put row values into their respective columns
        columns = zip(*data)
        
        if len(columns) == 1:   
            # Format data 
            col_list = list(columns[0])
            type = format[0]
            if type == 'i':
                col_list = [int(val) for val in col_list]
            elif type == 'f':
                col_list = [int(val) for val in col_list]
            elif type != 's':
                sys.exit("Warning: Unrecognized type " + type + " chosen!")    
            return col_list
        else:
            # Format data
            col_list = [list(tupl) for tupl in columns]
            type = format[0]
            if type == 'i':
                col_list = [map(int,col) for col in col_list]
            elif type == 'f':
                col_list = [map(float,col) for col in col_list]
            elif type != 's':
                sys.exit("Warning: Unrecognized type " + type + " chosen!")    
            return col_list 
        
    # Read a single column file
    # This requires the delimiter to be set to None, and col to be set to None
    # Only the first format character is used for formating, all others are 
    # ignored
    elif (delimiter == None) and (col == None):
        data = []
        for i,line in enumerate(infile.readlines()):
            if (i+1 in skiplines) == False: 
                field = line.strip()
                data.append(field)
                
        # Format data 
        type = format[0]
        if type == 'i':
            data = map(int,data)
        elif type == 'f':
            data = map(float,data)
        elif type != 's':
            sys.exit("Warning: Unrecognized type " + type + " chosen!")    
        return data
    
    # Read multicolumn file with different formats for the first N columns
    # where N is the number of format options chosen
    # This requires a delimiter to be set and col to be set to None
    elif (delimiter != None) and (col == None) and (len(format) > 1):
        data = [[] for dummy in range(len(format))]
        for i,line in enumerate(infile.readlines()):
            if (i+1 in skiplines) == False: 
                fields = line.strip().split(delimiter)
                fields = [field for field in fields if field != ""]
                for j,val in enumerate(fields):
                    try:
                        data[j].append(val)
                    except IndexError:
                        pass
    
    # Read multicolumn file with different formats for specific columns
    # This requires a delimiter to be set and col to be set
    # Col must be the same length as format 
    # *there is no difference between reading a single column with col set
    #  and having default col with no delimiter set 
    elif (delimiter != None) and (col != None):
        data = [[] for dummy in range(len(format))]
        assert(len(col) == len(format))
        for i,line in enumerate(infile.readlines()):
            if (i+1 in skiplines) == False: 
                fields = line.strip().split(delimiter)
                fields = [field for field in fields if field != ""]
                for j in range(len(col)):
                    try:
                        data[j].append(fields[col[j]-1])
                    except IndexError:
                        pass
    
    # Read 
    else:
        sys.exit("Error: Inappropriate input provide.")
                    
    infile.close()
        
    # Format data
    for i,type in enumerate(format):
        if type == 'i':
            data[i] = map(int,data[i])
        elif type == 'f':
            data[i] = map(float,data[i])
        elif type != 's':
            sys.exit("Warning: Unrecognized type " + type + " chosen!")
    
    # Convert to tuple 
    if len(data) == 1:
        columns = tuple(data[0])
    else: 
        columns = tuple(data)
    
    return columns

def write_flagfile(network_params):
    """
    Generates an LFR benchmark parameter file from chosen network parameters
    """

    # Make lines for each parameter
    lines = ""
    lines += '-N ' + str(network_params['N']) + '\n'
    lines += '-k ' + str(network_params['k']) + '\n'
    lines += '-maxk ' + str(network_params['maxk']) + '\n'
    lines += '-mu ' + str(network_params['mu']) + '\n'
    lines += '-t1 ' + str(network_params['t1']) + '\n'
    lines += '-t2 ' + str(network_params['t2']) + '\n'
    lines += '-minc ' + str(network_params['minc']) + '\n'
    lines += '-maxc ' + str(network_params['maxc']) + '\n'
    lines += '-on ' + str(network_params['on']) + '\n'
    lines += '-om ' + str(network_params['om']) + '\n'

    # Write-out to file
    flag_file = open(network_params['param_file'], 'w')
    flag_file.write(lines)
    flag_file.close()

def generate_graph(network_params, command_file, path):
    """
    Generates a graph from the LFR benchmark program with desired parameters
    """

    # Create parameter file
    write_flagfile(network_params)

    # Run program
    command = command_file + " -f " + network_params['param_file']
    os.popen(command)

    # Read output into networkx graph and return it
    LFR_graph = read_LFR_output(path + "network.dat", path + "community.dat")

    return LFR_graph

def read_LFR_output(edge_file, community_file):
    """
    Reads a LFR style output file into a networkx graph object
    """

    col1, col2, col3 = readcol(edge_file,'iif',delimiter='\t')
    col1 = np.array(col1)
    col2 = np.array(col2)
    edge_list = zip(col1, col2)
    
    nodes, clusters = readcol(community_file, 'ii',delimiter='\t')
    nodes = np.array(nodes)
    community_list = zip(nodes, clusters)

    # Create graph
    LFR_graph = nx.DiGraph(edge_list)

    # Assigne community values
    for node in LFR_graph.nodes():
        LFR_graph.node[node]['community'] = community_list[node-1][1]

    return LFR_graph

def make_lfr_graph(N, mu, k, maxk, minc, maxc, deg_exp=1.0, 
    com_exp=1.0, on=0, om=0, temp_dir_ID=0, full_path=None, benchmark_file=None):
    """
    Creates a temporary directory in which to generate an LFR graph, then removes the directory
    and returns a networkx graph
    """

    if full_path != None:
        path = full_path
        directory = path + "temp_" + str(temp_dir_ID) + "/"
        params={'N':N, 'k':k, 'maxk':maxk, 'mu':mu, 't1':deg_exp, 't2':com_exp, 
            'minc':minc, 'maxc':maxc, 'on':on, 'om':om, 'param_file':directory+'flags.dat'}

        if not os.path.exists(directory):
            os.makedirs(directory)
        if benchmark_file != None:
            command_file = path + benchmark_file
        else:
            command_file = path + "directed_benchmark"
        shutil.copy(command_file, directory)
        os.chdir(directory)

        graph = generate_graph(params, command_file, directory)

    else:
        path = './'
        directory = path + "temp_" + str(temp_dir_ID) + "/"
        params={'N':N, 'k':k, 'maxk':maxk, 'mu':mu, 't1':deg_exp, 't2':com_exp, 
            'minc':minc, 'maxc':maxc, 'on':on, 'om':om, 'param_file':path+'flags.dat'}

        if not os.path.exists(directory):
            os.makedirs(directory)
        if benchmark_file != None:
            command_file = path + benchmark_file
        else:
            command_file = path + "directed_benchmark"
        shutil.copy(command_file, directory)
        os.chdir(directory)

        graph = generate_graph(params, command_file, path)

    # move back
    if full_path:
        os.chdir(full_path)
    else:
        os.chdir('..')
    # del temp dir
    shutil.rmtree(directory)

    return graph

if __name__ == '__main__':
    """
    """
    
    nx.write_gexf(make_lfr_graph(5000, 0.1, 5, 5, 20, 20), 'test.gexf')