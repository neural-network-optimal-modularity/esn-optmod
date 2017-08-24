import networkx as nx
import numpy as np
import graphgen 
import utilities
import matplotlib.pyplot as plt
import Nbit_memory_objective
import subprocess
from sklearn.decomposition import PCA
import sys

class Nbit_task_analyzer(Nbit_memory_objective.Nbit_memory_objective):
    """
    """

    def __init__(self, **kwargs):
        """
        """
        super(Nbit_task_analyzer, self).__init__(**kwargs)

    def RandomInputRun(self, duration, num_trials):
        """
        """

        self.duration = duration
        self.num_trials = num_trials
        self.arrValTrials_arrTime_state = np.zeros((self.num_trials, self.duration, self.N))
        self.arrValTrials_arrTime_input = np.zeros((self.num_trials, self.duration, self.task.input_dimension))

        # create input w/ same dimesion as input_weights but ony 1's on just one channel
        multitrial_input_signal = np.zeros((self.num_trials, self.duration, self.task.input_dimension, 1))
        multitrial_input_signal[:,:,0,0] = np.ones(shape=(self.num_trials, self.duration))
        for i in xrange(self.num_trials):
            # Need to set the esn input weights to the randomly drawn set for the dummy task
            self.esn.input_weights = self.generate_input_weights()
            # Run esn
            for j in xrange(self.duration-1):
                self.esn.Step(multitrial_input_signal[i,j], record=True)

            self.arrValTrials_arrTime_state[i,:,:] = np.array(self.esn.network_history)[:,:,0]
            self.arrValTrials_arrTime_input[i,:,:] = multitrial_input_signal[i,:,:,0]
            self.esn.Reset()

    def ValidateModel(self, num_trials=1):
        """
        """

        if self.recall_task:
            test_input, test_targets = self.multitrial_input_signal, self.multitrial_target_signal
            num_trials = self.num_trials
        else:
            test_input, test_targets = self.generate_input(num_trials)

        # State, input, and output for all simulation time
        # self.num_trials = num_trials
        self.duration = self.task.total_duration
        self.arrValTrials_arrTime_state = np.zeros((num_trials, self.task.total_duration, self.N))
        self.arrValTrials_arrTime_input = np.zeros((num_trials, self.task.total_duration, self.task.input_dimension))
        self.arrValTrials_arrTime_output = np.zeros((num_trials, self.task.total_duration, self.sequence_dimension))
        self.arrValTrials_target = np.zeros((num_trials, self.sequence_length, self.sequence_dimension))
        self.arrValTrials_pred = np.zeros((num_trials, self.sequence_length, self.sequence_dimension))
        self.arrValTrials_performance = np.zeros((num_trials,1))

        self.inWeights = self.esn.input_weights.copy()
        self.outWeights = self.esn.output_weight_matrix.copy()
        self.reservoir = self.esn.reservoir.copy()

        for i in xrange(num_trials):
            performance, prediction, target_output, full_output = \
                self.esn.Predict(test_input[i], test_targets[i], 
                cut=0, recall_time=self.task.recall_time, 
                cut_target_output=False, target_range=(0,1), 
                error_type='NRMSE', analysis_mode=True)

            # Add data to arrays
            self.arrValTrials_arrTime_state[i,:,:] = np.array(self.esn.network_history)[:,:,0]
            self.arrValTrials_arrTime_input[i,:,:] = test_input[i,:,:,0]
            self.arrValTrials_arrTime_output[i,:,:] = full_output[:,:,0]
            self.arrValTrials_target[i,:,:] = target_output
            self.arrValTrials_pred[i,:,:] = prediction
            self.arrValTrials_performance[i] = performance

            self.esn.Reset()

        return 1.0 - np.count_nonzero(self.arrValTrials_performance) / float(len(self.arrValTrials_performance))#np.mean(self.arrValTrials_performance)

    def get_community_activity(self, trial_num):

        # Determine community activities
        dictCom_listNodes = utilities.get_community_node_dict(self.nx_reservoir, 'community')
        dictCom_listActivity = {}
        for com, nodes in dictCom_listNodes.items():
            # 1 is subtracted because the ids start from 1, while the indexes from 0
            community_state = self.arrValTrials_arrTime_state[trial_num,:,np.array(nodes)-1]
            dictCom_listActivity[com] = np.mean(community_state, axis=0)

        return dictCom_listActivity

    def plot_community_activity_timeseries(self, prefix, trial_num=0):
        """
        """

        dictCom_listActivity = self.get_community_activity(trial_num)

        # Plot activity over time
        for com, activity in dictCom_listActivity.items():
            # plt.clf()
            plt.plot(range(1,len(activity)+1), activity, lw=1, ls='-')

        # plt.axvline(self.start_time, color='black', ls='--', lw=2)
        plt.axvline(self.sequence_length, color='black', ls='--', lw=2, label='dis')
        plt.axvline(self.start_time + self.distraction_duration + self.sequence_length, 
            color='black', ls='-.', lw=2, label='cue')
        plt.axvline(self.task.recall_time,
            color='black', ls=':', lw=2, label='recall')
        plt.xlabel("time")
        plt.ylabel("activity")
        plt.legend(loc=2)
        plt.xlim(1,self.task.total_duration)
        plt.title('NRMSE: ' + str(round(self.arrValTrials_performance[trial_num], 4)), fontsize=10)
        plt.tight_layout()
        plt.savefig(prefix + "_com_activity_ts.png", dpi=300)
        plt.clf()
        plt.close()

    def plot_task_vs_com_activity(self, prefix, time="recall"):
        """
        """

        if time == "recall":
            time = self.task.recall_time
        else:
            time = time

        # Find unique tasks
        dictTask_arrInput = {}
        task_counter = 1
        for i, arr_input in enumerate(self.arrValTrials_target):
            if not check_if_in(arr_input, dictTask_arrInput.values()):
                # Add task 
                dictTask_arrInput[task_counter] = arr_input

                # # Create plot
                # dictCom_listActivity = self.get_community_activity(i)
                # plt.clf()
                # # print len(dictCom_listActivity.keys()), len(dictCom_listActivity.values()[time])
                # plt.bar(dictCom_listActivity.keys(), np.array(dictCom_listActivity.values())[:,time])
                # plt.xlim(0, len(dictCom_listActivity.keys()))
                # plt.tight_layout()
                # plt.savefig(prefix + "_task" + str(task_counter) + "_community_activity.png", dpi=300)
                plt.clf()
                plt.close()
                self.plot_community_activity_timeseries(prefix + "_task" + str(task_counter), i)
                task_counter += 1

    def plot_community_pca(self, prefix, dimensions=2, draw_trajectories=False):
        """
        """

        pass

    def plot_pca(self, prefix, dimensions=2, draw_trajectories=False):

        # Re-order data into an (# of conditions * # time-steps) X (# of neurons) data matrix [trajectory = True]
        if draw_trajectories:
            X = self.arrValTrials_arrTime_state.reshape((self.duration * self.num_trials, self.N))

        # Re-order data into an (# of conditions) X (# of neurons) data matrix [trajectory = False]
        else:
            X = self.arrValTrials_arrTime_state[:,-1,:]
            X = X.reshape((self.num_trials, self.N))

        # Apply PCA
        reservoirPCA = PCA(n_components=dimensions)
        reservoirPCA.fit(X)
        print(reservoirPCA.explained_variance_ratio_)

        # Plot PCA
        if dimensions == 2:
            if draw_trajectories == False:
                reduced_X = reservoirPCA.transform(X)
                plt.clf()
                plt.scatter(reduced_X[:,0], reduced_X[:,1])
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.savefig(prefix + "_pca_plot.png", dpi=300)
                plt.clf()
                plt.close()

            else:
                plt.clf()
                for i, trial in enumerate(xrange(self.num_trials)):
                    reduced_X = reservoirPCA.transform( self.arrValTrials_arrTime_state[i,:,:].reshape((self.duration, self.N)) )
                    plt.quiver(reduced_X[:-1, 0], reduced_X[:-1, 1], 
                        reduced_X[1:, 0]-reduced_X[:-1, 0], reduced_X[1:, 1]-reduced_X[:-1, 1], 
                        scale_units='xy', angles='xy', scale=1)

                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.savefig(prefix + "_traj-pca_plot.png", dpi=300)
                plt.clf()
                plt.close()

        if dimensions == 3:
            if draw_trajectories == False:
                from mpl_toolkits.mplot3d import Axes3D
                reduced_X = reservoirPCA.transform(X)
                plt.clf()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(reduced_X[:,0], reduced_X[:,1], reduced_X[:,2])
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
                plt.savefig(prefix + "_pca_3Dplot.png", dpi=300)
                plt.clf()
                plt.close()

            else:
                import scaled_quiver
                plt.clf()
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                for i, trial in enumerate(xrange(self.num_trials)):
                    reduced_X = reservoirPCA.transform( self.arrValTrials_arrTime_state[i,:,:].reshape((self.duration, self.N)) )
                    plt.quiver(reduced_X[:-1, 0], reduced_X[:-1, 1], reduced_X[:-1, 2],
                        reduced_X[1:, 0]-reduced_X[:-1, 0], 
                        reduced_X[1:, 1]-reduced_X[:-1, 1], 
                        reduced_X[1:, 2]-reduced_X[:-1, 2])

                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
                plt.savefig(prefix + "_traj-pca_3Dplot.png", dpi=300)
                plt.clf()
                plt.close()

def check_if_in(arr, list_arr):

    if len(list_arr) < 1:
        return False

    for arr0 in list_arr:
        if np.allclose(arr0, arr): 
            return True

    return False

def parameterize_taskmaster(parameters, q1_index, q2_index, full_path, num_validation_trials=None):

    parameters['parameters']['parameters']['nbit_task_parameters'][ parameters['parameters']['parameters']['q1'] ] = parameters['parameters']['q1_list'][q1_index]
    parameters['parameters']['parameters']['nbit_task_parameters'][ parameters['parameters']['parameters']['q2'] ] = parameters['parameters']['q2_list'][q2_index]
    parameters['parameters']['parameters']['nbit_task_parameters']['full_path'] = full_path
    test_parameter_set = parameters['parameters']['parameters']['nbit_task_parameters']
    return Nbit_task_analyzer(**test_parameter_set)

def get_validated_taskmaster(parameters, q1_index, q2_index, full_path, num_validation_trials=None):
    """
    """

    nbit_task_master = parameterize_taskmaster(parameters, q1_index, q2_index, full_path, num_validation_trials)
    nbit_task_master.TrainModel()

    if num_validation_trials == None:
        print 'NRMSE:', nbit_task_master.ValidateModel(parameters['parameters']['parameters']['num_validation_trials'])
    else:
        print 'NRMSE:', nbit_task_master.ValidateModel(num_validation_trials)

    return nbit_task_master

if __name__ == '__main__':
    """
    """

    params = {'backend': 'ps', \
              'axes.labelsize': 20, \
              'font.size': 20, \
              'legend.fontsize': 16, \
              'xtick.labelsize': 20, \
              'ytick.labelsize': 20, \
              'text.usetex': True, \
              'xtick.major.size': 10, \
              'ytick.major.size': 10, \
              'xtick.minor.size': 8, \
              'ytick.minor.size': 8, \
              'xtick.major.width': 1.0, \
              'ytick.major.width': 1.0, \
              'xtick.minor.width': 1.0, \
              'ytick.minor.width': 1.0}
    plt.rcParams.update(params)

    # Load results
    results = utilities.load_object('/home/nathaniel/workspace/DiscreteESN/nbit_N1k_dl4-5_recall_task/c1_e10_N1k_rsig0.3_gain2.0_ws1.0_lrb-0.1_urb1.0_dl4-5_mu-seq_final_results.pyobj')

    prefix = 'test'
    # Print results
    print(results['parameters'])
    # sys.exit()

    # Pick q1 and q1
    q1_index = 0
    q2_index = 7
    full_path = "/home/nathaniel/workspace/DiscreteESN/"

    # Train network in task
    nbit_task_master = get_validated_taskmaster(results, q1_index, q2_index, full_path)

    # Untrained random input response
    # nbit_task_master = parameterize_taskmaster(results, q1_index, q2_index, full_path)
    # nbit_task_master.RandomInputRun(200, 100)

    # Plot time series
    nbit_task_master.plot_task_vs_com_activity(prefix)

    # Do PCA
    # nbit_task_master.plot_pca(prefix, dimensions=3, draw_trajectories=True)
    # nbit_task_master.plot_pca(prefix, dimensions=3, draw_trajectories=False)