import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rnd
import utilities
import sys

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

    # Data (sem is over many reservoirs!, not over multiple trials. Just one trial that is very long)
    prefix = 'MC_N500_c1_e10_train1k_ws1.132_gani1.0_mu-select_rsig'
    final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    print(final_results['parameters'])
    i1, i2 = 11, 2
    listDelays_trials_mc1 = np.array([ [ final_results['results'][i1][i2][j][2][2][i] 
        for j in range(len(final_results['results'][i1][i2])) ]
        for i in range(len(final_results['results'][i1][i2][0][2][2])) ])

    print("MC", np.mean(np.array([ final_results['results'][i1][i2][j][2][0] 
        for j in range(len(final_results['results'][i1][i2])) ])))
    i1, i2 = 0, 2
    listDelays_trials_mc2 = np.array([ [ final_results['results'][i1][i2][j][2][2][i] 
        for j in range(len(final_results['results'][i1][i2])) ] 
        for i in range(len(final_results['results'][i1][i2][0][2][2])) ])
    delays = np.array(final_results['results'][i1][i2][0][2][1])

    print("MC", np.mean(np.array([ final_results['results'][i1][i2][j][2][0] 
        for j in range(len(final_results['results'][i1][i2])) ])))
    # Figure
    f, ax = plt.subplots(2, 2, figsize=(12,10))
    ax[0,0].axis('off')

    # Plot 1 MC curve
    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in listDelays_trials_mc1:
        list_means.append(np.mean(trials))
        sem = stats.sem(trials)
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    ax[0,1].errorbar(delays, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue', label='$\mu=0.28$')
    ax[0,1].fill_between(delays, 0, list_means, color='lightblue')
    ax[0,1].text(8.5, 0.5, '${MC}=10.1$', fontsize=20)

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in listDelays_trials_mc2:
        list_means.append(np.mean(trials))
        sem = stats.sem(trials)
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    ax[0,1].errorbar(delays, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='red', label='$\mu=0.0$')
    ax[0,1].fill_between(delays, 0, list_means, color='mistyrose')
    ax[0,1].text(2.5, 0.35, '${MC}=6.1$', fontsize=20)
    ax[0,1].legend(loc='upper right', frameon=False, fontsize=20)
    ax[0,1].set_ylabel("$r^2$", fontsize=30)
    ax[0,1].set_xlabel("delay", fontsize=20)
    ax[0,1].xaxis.set_ticks([4,8,12,16,20])
    ax[0,1].xaxis.labelpad = -10
    ax[0,1].xaxis.set_label_coords(0.47, -0.025)
    ax[0,1].set_ylim(0.0,1.0)
    ax[0,1].set_xlim(1,20)

    # Slice through contour
    listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in final_results['results']])
    listQ_listTrials_mc = listQ1_listQ2_listTrials_mc[:,2,:]
    q = final_results['parameters']['q1_list']

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in listQ_listTrials_mc:
        list_means.append(np.mean(trials))
        list_CIhigh.append(stats.sem(trials))
        list_CIlow.append(stats.sem(trials))

    ax[1,1].errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue')
    ax[1,1].set_ylabel("${MC}$", fontsize=20)
    ax[1,1].set_xlabel("$\mu$", fontsize=30)
    ax[1,1].set_ylim(ymin=0.0)
    ax[1,1].axvline(0.003, ls='-', lw=1.5, color='r')
    ax[1,1].axvline(0.289, ls='-', lw=1.5, color='b')

    # Contour plot
    prefix = 'MC_N500_c1_e10_train1k_gain1.0_ws1.132_mu-rsig'
    final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    print(final_results['parameters'])
    listQ1_listQ2_listTrials_mc = np.array([[[k[2][0] for k in j] for j in i] for i in final_results['results']])
    q1 = final_results['parameters']['q1_list']
    q2 = final_results['parameters']['q2_list']
    levels = np.linspace(4.0,11.0,8)

    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_mc in enumerate(listQ1_listQ2_listTrials_mc):
        for j, listTrials_mc in enumerate(listQ2_listTrials_mc):
            # print listTrials_nrmse, X[j,i], Y[j,i])
            Z[j,i] = np.mean(listTrials_mc)

    if levels==None:
        cont = ax[1,0].contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont = ax[1,0].contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar(cont, ax=ax[1,0])
    cbar.set_label("${MC}$")
    ax[1,0].set_xlabel("$\mu$", fontsize=30)
    ax[1,0].set_ylabel("$r_{\mathrm{sig}}$", fontsize=30)
    ax[1,0].axhline(0.3, ls='--', lw=3, color='w')


    plt.tight_layout(w_pad=-0.2, h_pad=0.4)
    plt.savefig("pres_mc_plot.png", dpi=900)