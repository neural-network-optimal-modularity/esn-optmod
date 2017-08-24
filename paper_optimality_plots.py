import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as rnd
import utilities

if __name__ == '__main__':
    
    params = {'backend': 'ps', 
              'axes.labelsize': 20,
              'text.fontsize': 20, 
              'legend.fontsize': 14,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
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
              'ps.fonttype' : 42
              }
    plt.rcParams.update(params)

    f, ax = plt.subplots(2, 2, figsize=(12,10))
    ratio_index = 7
    list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, \
        listRatios_listMus_listTrials_listResults = utilities.load_object("paper_version_rw0.4_iw3_is0.5_k6_simresults_vs_mu_vs_signal.pyobj")

    # Uses net FP index=12
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[12] /500.# divide by cirtuit size

    # [0,1]
    levels = np.linspace(0,1,11)
    if levels==None:
        cont=ax[0,1].contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont=ax[0,1].contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    # cbar = f.colorbar(cont,ax=ax[0,1])
    # cbar.set_label("\\text{Activity}")
    # ax[0,1].set_xlabel("$\mu$", fontsize=30)
    ax[0,1].set_ylabel("$r_{\mathrm{sig}}$", fontsize=30)
    ax[0,1].axhline(0.08, ls='--', lw=3, color='w')

    # Uses community FP
    listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    arrayMus_arrayTrials_com1activity = arrayMus_arrayTrials_arrayResults[:,:,13]
    arrayMus_arrayTrials_com2activity = arrayMus_arrayTrials_arrayResults[:,:,14]

    list_com1means = []
    list_com1CIhigh = []
    list_com1CIlow = []
    for trials in arrayMus_arrayTrials_com1activity:
        trials /= 500.# divide by cirtuit size
        sem = stats.sem(trials)
        list_com1means.append(np.mean(trials))
        list_com1CIhigh.append(sem)
        list_com1CIlow.append(sem)

    list_com2means = []
    list_com2CIhigh = []
    list_com2CIlow = []
    for trials in arrayMus_arrayTrials_com2activity:
        trials /= 500.# divide by cirtuit size
        sem = stats.sem(trials)
        list_com2means.append(np.mean(trials))
        list_com2CIhigh.append(sem)
        list_com2CIlow.append(sem)

    signal_ratio = list_signal_ratio[ratio_index]

    # [0,0]
    ax[0,0].errorbar(list_mus, list_com1means, yerr=[list_com1CIlow, list_com1CIhigh], marker='o', ls='-', color='blue', label='Seed community')
    ax[0,0].errorbar(list_mus, list_com2means, yerr=[list_com2CIlow, list_com2CIhigh], marker='o', ls='-', color='red', label='Neighboring Community')
    ax[0,0].set_ylim(0)
    ax[0,0].set_ylabel('\\text{Activity}')
    # ax[0,0].set_xlabel('$\mu$', fontsize=30)
    ax[0,0].legend(loc="upper right",frameon=False)

    # [1,1]
    list_mus, list_signal_ratio, listRatios_listMus_listMeanResults, \
        listRatios_listMus_listTrials_listResults = utilities.load_object("paper_50com_example_no_target_com_trials100_simresults_vs_mu_vs_signal.pyobj")
    # Uses net FP index=12
    num_com = int(listRatios_listMus_listMeanResults[0][0][2])
    X, Y = np.meshgrid(list_mus, list_signal_ratio)
    Z = np.zeros(shape=(len(list_signal_ratio), len(list_mus)))
    for i, listMus_listMeanResults in enumerate(listRatios_listMus_listMeanResults):
        for j, listMeanResults in enumerate(listMus_listMeanResults):
            Z[i,j] = listMeanResults[4+num_com+1+num_com+1+num_com] /500.# divide by cirtuit size
    levels = np.linspace(0,1,11)
    if levels==None:
        cont=ax[1,1].contourf(X, Y, Z, cmap=cm.magma)
    else:
        cont=ax[1,1].contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    # cbar = f.colorbar(cont,ax=ax[1,1])
    # cbar.set_label("\\text{Activity}")
    ax[1,1].set_xlabel("$\mu$", fontsize=30)
    ax[1,1].set_ylabel("$r_{\mathrm{sig}}$", fontsize=30)

    # Uses community FP
    ratio_index = 8 #5#6#8 #14


    # [1,0]
    ratio_index = 9
    val = np.linspace(0.05,0.3,20)[ratio_index]
    ax[1,1].axhline(val, ls='--', lw=3, color='white')#teal
    listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    arrayMus_arrayTrials_com1activity = arrayMus_arrayTrials_arrayResults[:,:,4+num_com+1+num_com+1+num_com] / 500.
    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for trials in arrayMus_arrayTrials_com1activity:
        sem = stats.sem(trials)
        list_means.append(np.mean(trials))
        list_CIhigh.append(sem)
        list_CIlow.append(sem)
    ax[1,0].errorbar(list_mus, list_means, yerr=[list_CIlow, list_CIhigh], color='black',
                marker='o', ls='-', label="$r_{\mathrm{sig}}" + "={v}$".format(v=str(round(val,2))))

    # ratio_index = 13
    # val = np.linspace(0.05,0.3,20)[ratio_index]
    # ax[1,1].axhline(val, ls='--', lw=3, color='olive')
    # listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    # arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    # arrayMus_arrayTrials_com1activity = arrayMus_arrayTrials_arrayResults[:,:,4+num_com+1+num_com+1+num_com] / 500.
    # list_means = []
    # list_CIhigh = []
    # list_CIlow = []
    # for trials in arrayMus_arrayTrials_com1activity:
    #     sem = stats.sem(trials)
    #     list_means.append(np.mean(trials))
    #     list_CIhigh.append(sem)
    #     list_CIlow.append(sem)
    # ax[1,0].errorbar(list_mus, list_means, yerr=[list_CIlow, list_CIhigh], color='olive',
    #             marker='None', ls='--', label="$r_{\mathrm{sig}}" + "={v}$".format(v=str(round(val,2))))

    # ratio_index = 16
    # val = np.linspace(0.05,0.3,20)[ratio_index]
    # ax[1,1].axhline(val, ls=':', lw=3, color='olive')
    # listMus_listTrials_listResults = listRatios_listMus_listTrials_listResults[ratio_index]
    # arrayMus_arrayTrials_arrayResults = np.array(listMus_listTrials_listResults)
    # arrayMus_arrayTrials_com1activity = arrayMus_arrayTrials_arrayResults[:,:,4+num_com+1+num_com+1+num_com] / 500.
    # list_means = []
    # list_CIhigh = []
    # list_CIlow = []
    # for trials in arrayMus_arrayTrials_com1activity:
    #     sem = stats.sem(trials)
    #     list_means.append(np.mean(trials))
    #     list_CIhigh.append(sem)
    #     list_CIlow.append(sem)
    # ax[1,0].errorbar(list_mus, list_means, yerr=[list_CIlow, list_CIhigh],
    #             marker='None', ls=':', color='olive',
    #             label="$r_{\mathrm{sig}}" + "={v}$".format(v=str(round(val,2))))

    ax[1,0].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax[1,0].set_ylim(0)
    ax[1,0].set_ylabel('\\text{Activity}')
    ax[1,0].set_xlabel('$\mu$', fontsize=30)
    ax[1,0].legend(loc="upper left",frameon=False)

    ax[0,0].text(-0.21, 1.0, '(b)', fontsize=25, fontname='Myriad Pro', transform = ax[0,0].transAxes) #[0,0]
    ax[0,1].text(1.05, 1.0, '(c)', fontsize=25, fontname='Myriad Pro', transform = ax[0,0].transAxes) #[0,1]
    ax[1,0].text(-0.21, -0.09, '(e)', fontsize=25, fontname='Myriad Pro', transform = ax[0,0].transAxes) #[1,0]
    ax[1,1].text(1.05, -0.09, '(f)', fontsize=25, fontname='Myriad Pro', transform = ax[0,0].transAxes) #[1,1]

    plt.tight_layout(w_pad=0.0, h_pad=0.0)
    f.subplots_adjust(right=0.89, wspace=0.325, hspace=0.08)
    cax = f.add_axes([0.91, 0.105, 0.02, 0.85])
    cbar = f.colorbar(cont, cax=cax)
    cbar.set_label("\\text{Activity}")


    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)

    plt.savefig("optimality_plots.svg", dpi=900)
    plt.close()
    plt.clf()