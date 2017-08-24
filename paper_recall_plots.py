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

    f, ax = plt.subplots(2, 2, figsize=(12,10))
    ax[0,0].axis('off')

    prefix = 'c1_e10_N1k_rsig0.3_gain2.0_ws1.0_lrb-0.1_urb1.0_dl4-5_mu-fp'
    results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    listQ1_listQ2_listTrials_numdiv = np.array([[[k[2][1] for k in j] for j in i] for i in results['results']])
    listQ1_listQ2_listTrials_numstable = np.array([[[k[2][0] for k in j] for j in i] for i in results['results']])
    listQ1_listTrials_numdiv = listQ1_listQ2_listTrials_numdiv[:,0,:]
    listQ1_listTrials_numstable = listQ1_listQ2_listTrials_numstable[:,0,:]
    listQ1_listTrials_numattractors = listQ1_listTrials_numdiv + listQ1_listTrials_numstable
    q = results['parameters']['q1_list']

    # list_means = []
    # list_CIhigh = []
    # list_CIlow = []
    # for trials in listQ1_listTrials_numstable:
    #     mean = np.mean(trials)
    #     list_means.append(mean)
    #     list_CIhigh.append(stats.sem(trials))
    #     list_CIlow.append(stats.sem(trials))

    # ax[0,1].errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
    #     marker='o', ls='-', color='blue')
    # # plt.plot(q, list_means, marker='o', ls='-', color='blue')
    # ax[0,1].set_ylabel("Number of Fixed Points", fontsize=20, color='b')
    # ax[0,1].set_xlabel("$\mu$", fontsize=30)
    # for tl in ax[0,1].get_yticklabels():
    #     tl.set_color('b')

    # list_means = []
    # list_CIhigh = []
    # list_CIlow = []
    # for trials in listQ1_listTrials_numdiv:
    #     mean = np.mean(trials)
    #     list_means.append(mean)
    #     list_CIhigh.append(stats.sem(trials))
    #     list_CIlow.append(stats.sem(trials))

    # ax2 = ax[0,1].twinx()
    # ax2.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
    #     marker='o', ls='-', color='red')
    # ax2.set_ylabel("Number non-stationary", fontsize=20, color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in listQ1_listTrials_numattractors:
        mean = np.mean(trials)
        list_means.append(mean)
        list_CIhigh.append(stats.sem(trials))
        list_CIlow.append(stats.sem(trials))

    ax[1,0].errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue')
    # plt.plot(q, list_means, marker='o', ls='-', color='blue')
    ax[1,0].set_ylabel("Number of attractors", fontsize=20, color='black')
    ax[1,0].set_xlabel("$\mu$", fontsize=30)
    ax[1,0].set_ylim(bottom=1)
    ax[1,0].set_ylim(bottom=0)

    prefix = 'c1_e10_N1k_rsig0.3_gain2_ws1_lrb-0.1_urb1.0_dl4-5_cue1_dis0_mu_dT'
    final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    listQ1_listQ2_listTrials_score = np.array([[[k[2][1] for k in j] for j in i] for i in final_results['results']])
    q1 = final_results['parameters']['q1_list']
    q2 = final_results['parameters']['q2_list']
    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_score in enumerate(listQ1_listQ2_listTrials_score):
        for j, listTrials_score in enumerate(listQ2_listTrials_score):
            Z[j,i] = np.mean(listTrials_score)

    levels = np.linspace(0,1,11)
    if levels==None:
        cont=ax[0,1].contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont=ax[0,1].contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    # cbar = plt.colorbar(cont, ax=ax[0,1])
    # cbar.set_label("score")
    # ax[0,1].set_xlabel("$\mu$", fontsize=30)
    ax[0,1].set_ylabel("$\Delta T$", fontsize=20)

    prefix = 'c1_e10_N1k_rsig0.3_gain2.0_ws1.0_lrb-0.1_urb1.0_dl4-5_mu-seq'
    final_results = utilities.load_object('/home/nathaniel/workspace/function_of_modularity/' + prefix + '_final_results.pyobj')
    listQ1_listQ2_listTrials_score = np.array([[[k[2][1] for k in j] for j in i] for i in final_results['results']])
    q1 = final_results['parameters']['q1_list']
    q2 = final_results['parameters']['q2_list']
    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_score in enumerate(listQ1_listQ2_listTrials_score):
        for j, listTrials_score in enumerate(listQ2_listTrials_score):
            Z[j,i] = np.mean(listTrials_score)

    levels = np.linspace(0,1,11)
    if levels==None:
        cont=ax[1,1].contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont=ax[1,1].contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    # cbar = plt.colorbar(cont, ax=ax[1,0])
    # cbar.set_label("score")
    ax[1,1].set_xlabel("$\mu$", fontsize=30)
    ax[1,1].set_ylabel("Number of sequences", fontsize=20)
    ax[1,1].axhline(200, ls='--', lw=3, color='w')

    plt.tight_layout(w_pad=0.0, h_pad=0.0)
    f.subplots_adjust(right=0.89, wspace=0.225, hspace=0.04)
    cax = f.add_axes([0.91, 0.105, 0.02, 0.85])
    cbar = f.colorbar(cont, cax=cax)
    cbar.set_label("\\text{score}")
    
    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)

    plt.savefig("paper_recall_plots.svg", dpi=900)