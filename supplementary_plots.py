import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rnd
import utilities
import sys

def spectral_figure():

    spectral_results = utilities.load_object("/home/nathaniel/workspace/function_of_modularity/MC_N500_ws1.132_com10_spectrum.pyobj")
    mu_array = np.linspace(0.0, 0.5, 20)

    first_mean = []
    first_sem = []
    for bytrial_byspectrum in spectral_results['adj_eigvals']:
        largest_by_trial = [ spectrum[-1] for spectrum in bytrial_byspectrum ]
        first_mean.append(np.mean(largest_by_trial))
        first_sem.append(stats.sem(largest_by_trial) * 1.97) # Turn SEM to ~95% interval

    second_mean = []
    second_sem = []
    for bytrial_byspectrum in spectral_results['adj_eigvals']:
        second_by_trial = [ spectrum[-2] for spectrum in bytrial_byspectrum ]
        second_mean.append(np.mean(second_by_trial))
        second_sem.append(stats.sem(second_by_trial) * 1.97) # Turn SEM to ~95% interval

    gap_mean = []
    gap_sem = []
    for bytrial_byspectrum in spectral_results['adj_eigvals']:
        gap_by_trial = [ spectrum[-1] - spectrum[-2] for spectrum in bytrial_byspectrum ]
        gap_mean.append(np.mean(gap_by_trial))
        gap_sem.append(stats.sem(gap_by_trial) * 1.97) # Turn SEM to ~95% interval

    fig, ax1 = plt.subplots()
    l1 = ax1.errorbar(mu_array, first_mean, yerr=[second_sem, second_sem],
    marker='o', ls='-', color='black')
    l2 = ax1.errorbar(mu_array, second_mean, yerr=[second_sem, second_sem],
        marker='o', ls='-', color='blue')
    ax2 = ax1.twinx()
    l3 = ax2.errorbar(mu_array, gap_mean, yerr=[gap_sem, gap_sem],
        marker='o', ls='--', color='red')

    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel("eigenvalue")
    ax2.set_ylabel("gap", color='r')
    ax2.tick_params('y', colors='r')
    fig.legend((l1, l2, l3), ('largest eigenvalue', 'second largest eigenvalue',
        'spectral gap'), loc="upper center", frameon=False, bbox_to_anchor=(0., 0.94, 1., 0))
    fig.tight_layout()
    plt.savefig("supp_spectrum.pdf")
    plt.close()
    plt.clf()

def ws_figure():

    ws_results = utilities.load_object("MC_N500_c1_e10_train1k_gain1_mu-ws_final_results.pyobj")
    # Contour plot
    print(ws_results['parameters'])
    listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in ws_results['results']])
    q1 = ws_results['parameters']['q1_list']
    q2 = ws_results['parameters']['q2_list']
    levels = [1,2,3,4,5,6,7,8,9,10,11]

    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_mc in enumerate(listQ1_listQ2_listTrials_mc):
        for j, listTrials_mc in enumerate(listQ2_listTrials_mc):
            # print listTrials_nrmse, X[j,i], Y[j,i])
            Z[j,i] = np.mean(listTrials_mc)

    if levels==None:
        cont = plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont = plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar(cont)
    cbar.set_label("${MC}$")
    plt.xlabel("$\mu$", fontsize=30)
    plt.ylabel("$W_s$", fontsize=30)
    plt.tight_layout()
    plt.savefig("supp_ws_figure.pdf")
    plt.close()
    plt.clf()

def spectral_rsig_figure():

    spectral_ws_results = utilities.load_object("const_spectral_MC_N500_c1_e10_train1k_ws1.132_gain1.0_com10_mu_v_rsig_final_results.pyobj")
    # f, ax = plt.subplots(1, 2, figsize=(12,5))

    # # Slice through contour
    # index1 = 7
    # listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in spectral_ws_results['results']])
    # listQ_listTrials_mc = listQ1_listQ2_listTrials_mc[:,index1,:]
    # q = spectral_ws_results['parameters']['q1_list']

    # list_means = []
    # list_CIhigh = []
    # list_CIlow = []
    # for trials in listQ_listTrials_mc:
    #     print(len(trials))
    #     list_means.append(np.mean(trials))
    #     list_CIhigh.append(stats.sem(trials))
    #     list_CIlow.append(stats.sem(trials))

    # ax[0].errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
    #     marker='o', ls='-', color='blue')
    # ax[0].set_ylabel("${MC}$", fontsize=20)
    # ax[0].set_xlabel("$\mu$", fontsize=30)
    # ax[0].set_ylim(ymin=0.0)

    # Contour plot
    print(spectral_ws_results['parameters'])
    listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in spectral_ws_results['results']])
    q1 = spectral_ws_results['parameters']['q1_list']
    q2 = spectral_ws_results['parameters']['q2_list']
    levels = np.linspace(4.0,11.0,8)

    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_mc in enumerate(listQ1_listQ2_listTrials_mc):
        for j, listTrials_mc in enumerate(listQ2_listTrials_mc):
            # print listTrials_nrmse, X[j,i], Y[j,i])
            Z[j,i] = np.mean(listTrials_mc)

    if levels==None:
        cont = plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont = plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar(cont)
    cbar.set_label("${MC}$")
    plt.xlabel("$\mu$", fontsize=30)
    plt.ylabel("$r_{\mathrm{sig}}$", fontsize=30)
    plt.tight_layout()
    plt.savefig("supp_spectral_rsig_figure.pdf")
    plt.close()
    plt.clf()

def overlap_figure():

    f, ax = plt.subplots(1, 2, figsize=(12,5))
    ax2 = ax[0].twinx()

    perf_results = utilities.load_object("/home/nathaniel/workspace/function_of_modularity/MC_N500_c1_e10_train1k_ws1.132_gain1.0_com10_mu_v_rsig_version2_final_results.pyobj")
    list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, \
        listRatios_listMus_listTrials_listResults = utilities.load_object("/home/nathaniel/workspace/function_of_modularity/N500_randinput_trials100_same_as_MC_task_simresults_vs_mu_vs_signal.pyobj")

    # Divide by duration and size of network to normalize
    oX, oY = np.meshgrid(list_mus, list_signal_ratio)
    oZ = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            oZ[i,j] = listMeanResults[3] /500./1000.# divide by cirtuit size and duration

    listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in perf_results['results']])
    q1 = perf_results['parameters']['q1_list']
    q2 = perf_results['parameters']['q2_list']
    levels = np.linspace(4.0,11.0,8)
    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_mc in enumerate(listQ1_listQ2_listTrials_mc):
        for j, listTrials_mc in enumerate(listQ2_listTrials_mc):
            Z[j,i] = np.mean(listTrials_mc)

    # Slice through contour
    index1 = 9
    listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in perf_results['results']])
    listQ_listTrials_mc = listQ1_listQ2_listTrials_mc[:,index1,:]
    q = perf_results['parameters']['q1_list']
    means = []
    sems = []
    for trials in listQ_listTrials_mc:
        means.append(np.mean(trials))
        sems.append(stats.sem(trials))
    l1 = ax[0].errorbar(q, means, yerr=[sems, sems], 
        marker='o', ls='-', color='blue')
    ax[0].set_ylabel("${MC}$", fontsize=20)
    ax[0].set_xlabel("$\mu$", fontsize=30)
    ax[0].set_ylim(ymin=0.0)

    means = []
    sems = []
    for trials in np.array(listRatios_listMus_listTrials_listResults)[index1,:,:,3]/500./1000.:# divide by cirtuit size and duration
        means.append(np.mean(trials))
        sems.append(stats.sem(trials))
    l2 = ax2.errorbar(q, means, yerr=[sems, sems], 
        marker='^', ls='--', color='red')
    ax2.set_ylabel("\\text{Activity}", fontsize=20, color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlabel("$\mu$", fontsize=30)
    ax2.set_ylim(ymin=0.0)

    # f.legend((l1, l2), ("${MC}$", 'Normalized activity'), 
    #     loc="lower center", frameon=False, bbox_to_anchor=(0., 0.5, 1., 0))

    if levels==None:
        cont = ax[1].contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont = ax[1].contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar(cont, ax=ax[1])
    cbar.set_label("${MC}$")
    cont2 = ax[1].contour(oX, oY, oZ, colors='k', linewidths=1.5)
    ax[1].clabel(cont2, fontsize=10, inline=1)
    ax[1].axhline(q2[index1], ls='--', lw=1., color='w')
    ax[1].set_xlabel("$\mu$", fontsize=30)
    ax[1].set_ylabel("$r_{\mathrm{sig}}$", fontsize=30)

    ax[0].text(-0.15, 1.0, '(a)', fontsize=25, fontname='Myriad Pro', transform = ax[0].transAxes)
    ax[1].text(-0.25, 1.0, '(b)', fontsize=25, fontname='Myriad Pro', transform = ax[1].transAxes)

    plt.tight_layout()
    plt.savefig("supp_mc_optmod_contour.pdf")
    plt.close()
    plt.clf()


if __name__ == '__main__':

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

    ##########################################################################
    # Spectral figure
    ##########################################################################
    # spectral_figure()

    ##########################################################################
    # WS figure
    ##########################################################################
    # ws_figure()

    ##########################################################################
    # Controlled largest spectral radius figure
    ##########################################################################
    # spectral_rsig_figure()

    ##########################################################################
    # Opt/perf overlap
    ##########################################################################
    overlap_figure()