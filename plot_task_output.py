import numpy as np 
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import bayes_mvs
import utilities
from scipy.stats import sem

def contour_plot(prefix, q1, q2, listQ1_listQ2_listTrials_val, 
    xlabel, ylabel, logx=False, logy=False, logz=False, levels=None, zlabel=None):

    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_val in enumerate(listQ1_listQ2_listTrials_val):
        for j, listTrials_val in enumerate(listQ2_listTrials_val):
            Z[j,i] = np.mean(listTrials_val)

    plt.clf()
    # fig, ax = plt.subplots()
    # ax.ticklabel_format(useOffset=False,style='plain')
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    if zlabel:
        cbar.set_label(zlabel)
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if logz:
        plt.zscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_q1_q2_contour.png')
    plt.clf()
    plt.close()

def performance_contour_plot(prefix, q1, q2, listQ1_listQ2_listTrials_nrmse, 
    xlabel, ylabel, zlabel, logx=False, logy=False, logz=False, levels="auto"):

    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_nrmse in enumerate(listQ1_listQ2_listTrials_nrmse):
        for j, listTrials_nrmse in enumerate(listQ2_listTrials_nrmse):
            # print listTrials_nrmse, X[j,i], Y[j,i])
            Z[j,i] = np.mean(listTrials_nrmse)

    plt.clf()
    # fig, ax = plt.subplots()
    # ax.ticklabel_format(useOffset=False,style='plain')
    if levels is "auto":
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label(zlabel) #"${NRMSE}$"
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if logz:
        plt.zscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_q1_q2_mc_contour.png')
    plt.clf()
    plt.close()

def performance_vs_q_plot(prefix, q, listQ_listTrials_nrmse, xlabel, logx=False, logy=False):

    mean_err = sem(listQ_listTrials_nrmse, axis=1)
    mean = np.mean(listQ_listTrials_nrmse, axis=1)
    high = mean_err
    low = mean_err

    plt.errorbar(q, mean, yerr=[mean_err, mean_err], 
        marker='o', ls='-', color='blue')
    # plt.plot(q, list_means, marker='o', ls='-', color='blue')
    plt.ylabel("${MC}$", fontsize=30) #"${NRMSE}$"
    plt.xlabel(xlabel, fontsize=30)
    plt.ylim(ymin=0.0)
    plt.tight_layout()
    plt.savefig(prefix + "_mc_vs_q.png", dpi=300)
    plt.close()
    plt.clf()

def overlay_contour_plot(prefix, q1, q2, lead_q1_q2_trials_dyn, follows_q1_q2_trials_nrmse, 
    xlabel, ylabel, logx=False, logy=False, logz=False, levels="auto", levels2="auto"):
    """
    """

    # Lead contour
    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_nrmse in enumerate(lead_q1_q2_trials_dyn):
        for j, listTrials_nrmse in enumerate(listQ2_listTrials_nrmse):
            Z[j,i] = np.mean(listTrials_nrmse)

    plt.clf()
    # fig, ax = plt.subplots()
    # ax.ticklabel_format(useOffset=False,style='plain')
    if levels is "auto":
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    # cbar = plt.colorbar()
    # cbar.set_label('${NRMSE}$')

    color_list = ['r', 'b', 'g']
    for l, follow in enumerate(follows_q1_q2_trials_nrmse):
        Z = np.zeros((len(q2),len(q1)))
        for i, listQ2_listTrials_nrmse in enumerate(follow):
            for j, listTrials_nrmse in enumerate(listQ2_listTrials_nrmse):
                Z[j,i] = np.mean(listTrials_nrmse)

        if levels2 is "auto":
            CS = plt.contour(X, Y, Z, colors=color_list[l])
        else:
            CS = plt.contour(X, Y, Z, colors=color_list[l], levels=levels2)
        plt.clabel(CS, fontsize=10, inline=1)

    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if logz:
        plt.zscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_q1_q2_nrmse_contour.png')
    plt.clf()
    plt.close()

def plot_reservoir_stability(prefix, q, q1_trails_numfp, q1_trails_numdiv):
    """
    """

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in q1_trails_numfp:
        mean, var, std = utilities.bayes_mvs_wrapper(trials, alpha=0.95)
        list_means.append(mean[0])
        list_CIhigh.append(std[0])
        list_CIlow.append(std[0])

    fig, ax = plt.subplots()
    ax.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue')
    # plt.plot(q, list_means, marker='o', ls='-', color='blue')
    ax.set_ylabel("Number of Fixed Points", fontsize=20, color='b')
    ax.set_xlabel("$\mu$", fontsize=30)
    for tl in ax.get_yticklabels():
        tl.set_color('b')

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in q1_trails_numdiv:
        mean, var, std = utilities.bayes_mvs_wrapper(trials, alpha=0.95)
        list_means.append(mean[0])
        list_CIhigh.append(std[0])
        list_CIlow.append(std[0])

    ax2 = ax.twinx()
    ax2.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='red')
    ax2.set_ylabel("Number non-stationary", fontsize=20, color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax.set_ylim(bottom=1)
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(prefix + "_numfp_vs_mu.png", dpi=300)
    plt.close()
    plt.clf()

def plot_reservoir_transients(prefix, q, q_trials_translengths, q_trials_numnonstationary):

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in q_trials_translengths:
        unfolded_trials = trials.flatten()
        unfolded_trials = unfolded_trials[~np.isnan(unfolded_trials)]
        if len(unfolded_trials) == 0:
            list_means.append(np.nan)
            list_CIhigh.append(np.nan)
            list_CIlow.append(np.nan)

        mean, var, std = utilities.bayes_mvs_wrapper(unfolded_trials, alpha=0.95)
        list_means.append(mean[0])
        list_CIhigh.append(std[0])
        list_CIlow.append(std[0])

    fig, ax = plt.subplots()
    ax.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue')
    # plt.plot(q, list_means, marker='o', ls='-', color='blue')
    ax.set_ylabel("Transient length", fontsize=20, color='b')
    ax.set_xlabel("$\mu$", fontsize=30)
    for tl in ax.get_yticklabels():
        tl.set_color('b')

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in q_trials_numnonstationary:
        mean, var, std = utilities.bayes_mvs_wrapper(trials, alpha=0.95)
        list_means.append(mean[0])
        list_CIhigh.append(std[0])
        list_CIlow.append(std[0])

    ax2 = ax.twinx()
    ax2.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='red')
    ax2.set_ylabel("Number non-stationary", fontsize=20, color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    ax.set_ylim(bottom=1)
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(prefix + "_transient_vs_mu.png", dpi=300)
    plt.close()
    plt.clf()

def plot_mc_curves(prefix, listDelays_trials_mc, delays):

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in listDelays_trials_mc:
        mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
        list_means.append(mean[0])
        list_CIhigh.append(mean[1][1] - mean[0])
        list_CIlow.append(mean[0] - mean[1][0])

    print len(delays), len(list_means)
    plt.errorbar(delays, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue')

    # plt.plot(delays, list_means, marker='o', ls='-', color='blue')
    plt.ylabel("$r^2$", fontsize=30)
    plt.xlabel("delay", fontsize=30)
    plt.tight_layout()
    plt.ylim(0.0,1.0)
    plt.savefig(prefix + "_MC_vs_delay.png", dpi=300)
    plt.close()
    plt.clf()

def plot_separability_curves(prefix, q1, list_euclidean_dst, listQ1_listSamples_listDst_listTrials,
    listQ1_listSamples_MC):

    max_mc = np.max(listQ1_listSamples_MC)
    for i, q in enumerate(q1):
        arDst_arTrials = listQ1_listSamples_listDst_listTrials[i].reshape(listQ1_listSamples_listDst_listTrials.shape[2], -1)
        norm_mc = np.mean(listQ1_listSamples_MC[i]) / max_mc 
        # high = np.percentile(arDst_arTrials, 95, axis=1) - np.mean(arDst_arTrials, axis=1)
        # low = np.mean(arDst_arTrials, axis=1) - np.percentile(arDst_arTrials, 5, axis=1)
        mean_err = sem(arDst_arTrials, axis=1)
        mean = np.mean(arDst_arTrials, axis=1)
        high = mean_err
        low = mean_err
        plt.errorbar(list_euclidean_dst, np.mean(arDst_arTrials, axis=1), 
            yerr=[low, high], label="$\mu=$" + str(q), c=cm.magma(norm_mc))

    plt.legend(loc="upper left")
    plt.xlabel("input distance")
    plt.ylabel("state distance")
    plt.tight_layout()
    plt.savefig(prefix + "_separability_curve.png", dpi=300)
    plt.clf()
    plt.close()

def plot_separability_curves_by_reservoir(prefix, q1, list_euclidean_dst, listQ1_listSamples_listDst_listTrials):

    for i, q in enumerate(q1):
        for j in range(listQ1_listSamples_listDst_listTrials.shape[1]):
            arDst_arTrials = listQ1_listSamples_listDst_listTrials[i,j,:,:]
            # high = np.percentile(arDst_arTrials, 95, axis=1) - np.mean(arDst_arTrials, axis=1)
            # low = np.mean(arDst_arTrials, axis=1) - np.percentile(arDst_arTrials, 5, axis=1)
            mean_err = sem(arDst_arTrials, axis=1)
            mean = np.mean(arDst_arTrials, axis=1)
            high = mean_err
            low = mean_err
            plt.errorbar(list_euclidean_dst, np.mean(arDst_arTrials, axis=1), yerr=[low, high], label="$\mu=$" + str(q))

            plt.legend(loc="upper left")
            plt.xlabel("input distance")
            plt.ylabel("state distance")
            plt.tight_layout()
            plt.savefig(prefix + "_separability_res" +str(j) + "_mu" + str(i) + ".png", dpi=300)
            plt.clf()
            plt.close()

def plot_separability_performance_curves_by_reservoir(prefix, q1, list_euclidean_dst, 
    listQ1_listSamples_listDst_listTrials, listQ1_listSamples_MC):

    max_mc = np.max(listQ1_listSamples_MC)
    for i, q in enumerate(q1):
        for j in range(listQ1_listSamples_listDst_listTrials.shape[1]):
            arDst_arTrials = listQ1_listSamples_listDst_listTrials[i,j,:,:]
            norm_mc = listQ1_listSamples_MC[i,j] / max_mc
            # high = np.percentile(arDst_arTrials, 95, axis=1) - np.mean(arDst_arTrials, axis=1)
            # low = np.mean(arDst_arTrials, axis=1) - np.percentile(arDst_arTrials, 5, axis=1)
            mean_err = sem(arDst_arTrials, axis=1)
            mean = np.mean(arDst_arTrials, axis=1)
            high = mean_err
            low = mean_err
            plt.errorbar(list_euclidean_dst, np.mean(arDst_arTrials, axis=1), yerr=[low, high], 
                c=cm.magma(norm_mc))

        plt.legend(loc="upper left")
        plt.xlabel("input distance")
        plt.ylabel("state distance")
        plt.xlim(0,list_euclidean_dst[-1])
        plt.ylim(0,5)
        plt.tight_layout()
        plt.savefig(prefix + "_separability_performance_mu" + str(q) + ".png", dpi=300)
        plt.clf()
        plt.close()


if __name__ == '__main__':
    """
    """
    params = {'backend': 'ps', 
              'axes.labelsize': 20,
              'text.fontsize': 20, 
              'legend.fontsize': 14,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
              'text.usetex': True, 
              'xtick.major.size': 10,
              'ytick.major.size': 10,
              'xtick.minor.size': 8, 
              'ytick.minor.size': 8, 
              'xtick.major.width': 1.0, 
              'ytick.major.width': 1.0, 
              'xtick.minor.width': 1.0, 
              'ytick.minor.width': 1.0,
              'pdf.fonttype' : 42,
              'ps.fonttype' : 42}
    plt.rcParams.update(params)

    prefix = 'MC_N500_c1_e10_train1k_ws1.132_gain1.0_com10_mu_v_rsig_select'
    final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    print(final_results['parameters'])

    # # Contour plot
    # # final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    # listQ1_listQ2_listTrials_nrmse = np.array([[[k[2][0] for k in j] for j in i] for i in final_results['results']])
    # # listQ1_listQ2_listTrials_nrmse = np.array(final_results['results'])[:,:,:,2]
    # performance_contour_plot(prefix, final_results['parameters']['q1_list'], 
    #     final_results['parameters']['q2_list'], listQ1_listQ2_listTrials_nrmse,
    #     r'$\mu$',r'$r_{\textup{sig}}$', 'Score', logx=False, logy=False, logz=False)#,, levels=np.linspace(0.0,20.0,21) )#r'${urb}$', r'${lrb}$'

    # MC curve
    # final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    # print(final_results['parameters'])
    # i1, i2 = 13, 2
    # listDelays_Trials_MC = np.array([ [ final_results['results'][i1][i2][j][2][2][i] for j in range(len(final_results['results'][i1][i2])) ] 
    #     for i in range(len(final_results['results'][i1][i2][0][2][2])) ])
    # delays = np.array(final_results['results'][i1][i2][0][2][1])
    # plot_mc_curves(prefix, listDelays_Trials_MC, delays)

    # # 1D Plot
    # final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    # print(final_results['parameters'])
    # listQ_listTrials_nrmse = np.array(final_results['results'])[:,0,:,2]
    listQ1_listQ2_listTrials_nrmse = np.array([[[k[2][0] for k in j] for j in i] for i in final_results['results']])
    # print listQ1_listQ2_listTrials_nrmse
    # print listQ1_listQ2_listTrials_nrmse[0][0]
    listQ_listTrials_nrmse = listQ1_listQ2_listTrials_nrmse[:,0,:]
    performance_vs_q_plot(prefix+"_1", final_results['parameters']['q1_list'], listQ_listTrials_nrmse, r'$\mu$')

    listQ_listTrials_nrmse = listQ1_listQ2_listTrials_nrmse[:,1,:]
    performance_vs_q_plot(prefix+"_2", final_results['parameters']['q1_list'], listQ_listTrials_nrmse, r'$\mu$')

    listQ_listTrials_nrmse = listQ1_listQ2_listTrials_nrmse[:,2,:]
    performance_vs_q_plot(prefix+"_3", final_results['parameters']['q1_list'], listQ_listTrials_nrmse, r'$\mu$')

    listQ_listTrials_nrmse = listQ1_listQ2_listTrials_nrmse[:,3,:]
    performance_vs_q_plot(prefix+"_4", final_results['parameters']['q1_list'], listQ_listTrials_nrmse, r'$\mu$')

    # # Overlay plot
    # # proc_results = utilities.load_object('./_final_results.pyobj')
    # mem_results = utilities.load_object('./binary_memory_objective_mu-rsig2_final_results.pyobj')
    # dyn_results = utilities.load_object('./binary_null_objective_mu-rsig2_final_results.pyobj')

    # # proc_listQ1_listQ2_listTrials_nrmse = np.array(proc_results['results'])[:,:,:,2]
    # mem_listQ1_listQ2_listTrials_nrmse = np.array(mem_results['results'])[:,:,:,2]
    # dyn_listQ1_listQ2_listTrials_act = np.array([[[k[2][0] for k in j] for j in i] for i in dyn_results['results']])

    # overlay_contour_plot("test", dyn_results['parameters']['list_q1'], dyn_results['parameters']['q2_list'], 
    #     dyn_listQ1_listQ2_listTrials_act, (mem_listQ1_listQ2_listTrials_nrmse,), 
    #     r'$\mu$', r'$r_{sig}$', logx=False, logy=False, logz=False, levels2=[0.55, 0.6, 0.65])

    # # Dynamical plots
    # prefix = 'fp_ic0-0.8_tmax1k_N500_c1_e10_train1k_ws1.132_mu'
    # results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    # listQ1_listQ2_listTrials_numdiv = np.array([[[k[2][1] for k in j] for j in i] for i in results['results']])
    # listQ1_listQ2_listTrials_numstable = np.array([[[k[2][0] for k in j] for j in i] for i in results['results']])

    # listQ1_listTrials_numdiv = listQ1_listQ2_listTrials_numdiv[:,0,:]
    # listQ1_listTrials_numstable = listQ1_listQ2_listTrials_numstable[:,0,:]
    
    # plot_reservoir_stability(prefix, results['parameters']['q1_list'], listQ1_listTrials_numstable, listQ1_listTrials_numdiv)

    # # Dynamic contour
    # results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/fp_mu_ws0.75_final_results.pyobj')
    # listQ1_listQ2_listTrials_numdiv = np.array([[[k[2][1] for k in j] for j in i] for i in results['results']])
    # listQ1_listQ2_listTrials_numstable = np.array([[[k[2][0] for k in j] for j in i] for i in results['results']])
    # contour_plot("fp_mu_ws0.75_div", results['parameters']['q1_list'], results['parameters']['q2_list'], listQ1_listQ2_listTrials_numdiv,
    #     r'$\mu$', r'$W_s$')

    # # Transient plots
    # prefix = 'trans_N1k_c1_e10_ws1_IC0-1_mu'
    # results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    # listQ1_listQ2_listTrials_listTranslength = np.array([[[k[2][0] for k in j] for j in i] for i in results['results']])
    # listQ1_listQ2_listTrials_numnonstationary = np.array([[[k[2][1] for k in j] for j in i] for i in results['results']])
    # listQ1_listQ2_listTrials_listTranslength = listQ1_listQ2_listTrials_listTranslength[:,0,:]
    # listQ1_listQ2_listTrials_numnonstationary = listQ1_listQ2_listTrials_numnonstationary[:,0,:]
    # plot_reservoir_transients(prefix, results['parameters']['q1_list'], 
    #     listQ1_listQ2_listTrials_listTranslength, listQ1_listQ2_listTrials_numnonstationary)

    # # Separability plots
    # prefix = 'debug_perf_sep_N500_c1_e10_ws1.132_rsig0.3_mu'
    # results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    # listQ1_listQ2_listSamples_listDst_listTrials = np.array([[[k[2] for k in j] for j in i] for i in results['results']])
    # listQ1_listQ2_listSamples_MC = np.array([[[k[4][0] for k in j] for j in i] for i in results['results']])
    # listQ1_listSamples_listDst_listTrials = listQ1_listQ2_listSamples_listDst_listTrials[:,0,:,:]
    # listQ1_listSamples_MC = listQ1_listQ2_listSamples_MC[:,0,:]
    # list_euclidean_dst = np.array([[[k[3] for k in j] for j in i] for i in results['results']])[0,0,0]
    # plot_separability_curves(prefix, results['parameters']['q1_list'], list_euclidean_dst, 
    #     listQ1_listSamples_listDst_listTrials, listQ1_listSamples_MC)
    # plot_separability_performance_curves_by_reservoir(prefix, results['parameters']['q1_list'], 
    #     list_euclidean_dst, listQ1_listSamples_listDst_listTrials, listQ1_listSamples_MC)
