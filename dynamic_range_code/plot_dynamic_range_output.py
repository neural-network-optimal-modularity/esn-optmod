import numpy as np 
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import bayes_mvs
import utilities
import sys

# def contour_plot(prefix, q1, q2, listQ1_listQ2_listTrials_val, 
#     xlabel, ylabel, logx=False, logy=False, logz=False, levels=None, zlabel=None):

#     X, Y = np.meshgrid(q1, q2)
#     Z = np.zeros((len(q2),len(q1)))
#     for i, listQ2_listTrials_val in enumerate(listQ1_listQ2_listTrials_val):
#         for j, listTrials_val in enumerate(listQ2_listTrials_val):
#             Z[j,i] = np.mean(listTrials_val)

#     plt.clf()
#     # fig, ax = plt.subplots()
#     # ax.ticklabel_format(useOffset=False,style='plain')
#     if levels==None:
#         plt.contourf(X, Y, Z, cmap=cm.magma)
#     else:
#         plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
#     cbar = plt.colorbar()
#     if zlabel:
#         cbar.set_label(zlabel)
#     plt.xlabel(xlabel, fontsize=30)
#     plt.ylabel(ylabel, fontsize=30)
#     if logx:
#         plt.xscale('log')
#     if logy:
#         plt.yscale('log')
#     if logz:
#         plt.zscale('log')

#     plt.tight_layout()
#     plt.savefig(prefix + '_q1_q2_contour.png')
#     plt.clf()
#     plt.close()

# def performance_contour_plot(prefix, q1, q2, listQ1_listQ2_listTrials_nrmse, 
#     xlabel, ylabel, logx=False, logy=False, logz=False, levels=None):

#     X, Y = np.meshgrid(q1, q2)
#     Z = np.zeros((len(q2),len(q1)))
#     for i, listQ2_listTrials_nrmse in enumerate(listQ1_listQ2_listTrials_nrmse):
#         for j, listTrials_nrmse in enumerate(listQ2_listTrials_nrmse):
#             # print listTrials_nrmse, X[j,i], Y[j,i])
#             Z[j,i] = np.mean(listTrials_nrmse)

#     plt.clf()
#     # fig, ax = plt.subplots()
#     # ax.ticklabel_format(useOffset=False,style='plain')
#     if levels==None:
#         plt.contourf(X, Y, Z, cmap=cm.magma)
#     else:
#         plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
#     cbar = plt.colorbar()
#     cbar.set_label('${Score}$') #"${NRMSE}$"
#     plt.xlabel(xlabel, fontsize=30)
#     plt.ylabel(ylabel, fontsize=30)
#     if logx:
#         plt.xscale('log')
#     if logy:
#         plt.yscale('log')
#     if logz:
#         plt.zscale('log')

#     plt.tight_layout()
#     plt.savefig(prefix + '_q1_q2_nrmse_contour.png')
#     plt.clf()
#     plt.close()

# def performance_vs_q_plot(prefix, q, listQ_listTrials_nrmse, xlabel, logx=False, logy=False):

#     list_means = []
#     list_CIhigh = []
#     list_CIlow = []
#     for trials in listQ_listTrials_nrmse:
#         mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
#         list_means.append(mean[0])
#         list_CIhigh.append(mean[1][1] - mean[0])
#         list_CIlow.append(mean[0] - mean[1][0])

#     plt.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
#         marker='o', ls='-', color='blue')
#     # plt.plot(q, list_means, marker='o', ls='-', color='blue')
#     plt.ylabel("${Score}$", fontsize=30) #"${NRMSE}$"
#     plt.xlabel(xlabel, fontsize=30)
#     plt.ylim(ymin=0.0)
#     plt.tight_layout()
#     plt.savefig(prefix + "_nrmse_vs_q.png", dpi=300)
#     plt.close()
#     plt.clf()

# def plot_mc_curves(prefix, listDelays_trials_mc, delays):

#     list_means = []
#     list_CIhigh = []
#     list_CIlow = []
#     for trials in listDelays_trials_mc:
#         mean, var, std = bayes_mvs_wrapper(trials, alpha=0.95)
#         list_means.append(mean[0])
#         list_CIhigh.append(mean[1][1] - mean[0])
#         list_CIlow.append(mean[0] - mean[1][0])

#     print len(delays), len(list_means)
#     plt.errorbar(delays, list_means, yerr=[list_CIlow, list_CIhigh], 
#         marker='o', ls='-', color='blue')

#     # plt.plot(delays, list_means, marker='o', ls='-', color='blue')
#     plt.ylabel("$r^2$", fontsize=30)
#     plt.xlabel("delay", fontsize=30)
#     plt.tight_layout()
#     plt.ylim(0.0,1.0)
#     plt.savefig(prefix + "_MC_vs_delay.png", dpi=300)
#     plt.close()
#     plt.clf()

def calc_dynamical_range(Arates_p, Arates_F):
    """
    determines the dynamical range based on 10% and 90% of response
    """

    from scipy.interpolate import interp1d

    interpol = interp1d(Arates_F, Arates_p, kind='linear')
    fmin = Arates_F.min()
    fmax = Arates_F.max()
    f1 = fmin + 0.1 * (fmax - fmin)
    f9 = fmin + 0.9 * (fmax - fmin)

    p1 = interpol(f1)
    p9 = interpol(f9)

    # xnew = np.linspace(min(Arates_F),max(Arates_F))
    # print p1, p9
    # plt.plot(Arates_F, Arates_p , 'o', xnew, interpol(xnew), '-', xnew, interpol(xnew), '--')
    # plt.show()
    # print 10 * np.log10(p9 / p1), p1, p9

    if (p1 < 0) or (p9 > 1.0):
        print "Warning: p1 or p9 exceed bounds", p1, p9

    return 10 * np.log10(p9 / p1), p1, p9

def plot_mc_vs_p(prefix, Arates_p, Arates_MC):
    """
    plot the performance versus the binomial success rate
    """

    plt.plot(Arates_p, Arates_MC)
    plt.xlabel("$\lambda$")
    plt.ylabel("MC")
    plt.tight_layout()
    plt.savefig(prefix + "_mc_vs_p.png", dpi=300)
    plt.clf()
    plt.close()

def plot_F_vs_p(prefix, Arates_p, Arates_F):
    """
    plot the activity versus the binomial success rate
    """

    _, p1, p9 = calc_dynamical_range(Arates_p, Arates_F)
    plt.plot(Arates_p, Arates_F)
    plt.axvline(p1, color='black', ls='--')
    plt.axvline(p9, color='black', ls='--')
    plt.xlabel("$\lambda$")
    plt.ylabel("activity")
    plt.tight_layout()
    plt.savefig(prefix + "_F_vs_p.png", dpi=300)
    plt.clf()
    plt.close()

def plot_dynrange_vs_q(prefix, Qs, Aq_Asample_Arates_p, Aq_Asample_Arates_F):
    """
    plot the dynamic range averaged over the number of sample graphs versus a single variable
    """

    # Calculate dynamic range
    dynrange_by_q = []
    CIlow_by_q = []
    CIhigh_by_q = []
    for i in range(Aq_Asample_Arates_F.shape[0]):
        sample_ranges = []
        for j in range(Aq_Asample_Arates_F.shape[1]):
            dynrange, _, _ = calc_dynamical_range(Aq_Asample_Arates_p[i, j], Aq_Asample_Arates_F[i,j])
            sample_ranges.append(dynrange)

        mean, var, std = utilities.bayes_mvs_wrapper(sample_ranges, alpha=0.95)
        dynrange_by_q.append(mean[0])
        CIlow_by_q.append(mean[1][1] - mean[0])
        CIhigh_by_q.append(mean[0] - mean[1][0])
        
    plt.errorbar(Qs, dynrange_by_q, yerr=[CIlow_by_q, CIhigh_by_q], 
        marker='o', ls='-', color='blue')
    plt.ylabel("$\Delta$")
    plt.xlabel("$\mu$")
    plt.tight_layout()
    plt.savefig(prefix + "_delta_vs_mu.png", dpi=300)
    plt.clf()
    plt.close()

def plot_dynrange_contour(prefix, Q1, Q2, Aq1_Aq2_Asample_Arates_p, Aq1_Aq2_Asample_Arates_F,
    xlabel, ylabel, logx=False, logy=False, logz=False, levels=None):
    """
    plot the dynamic rage averaged over a number of sample graphs versus two variables
    """

    X, Y = np.meshgrid(Q1, Q2)
    Z = np.zeros((len(Q2),len(Q1)))
    for i in range(Aq1_Aq2_Asample_Arates_p.shape[0]):
        for j in range(Aq1_Aq2_Asample_Arates_p.shape[1]):
            dynranges = []
            for k in range(Aq1_Aq2_Asample_Arates_p.shape[2]):
                dynrange, _, _ = calc_dynamical_range(Aq1_Aq2_Asample_Arates_p[i, j, k], Aq1_Aq2_Asample_Arates_F[i,j, k])
                dynranges.append(dynrange)

            Z[j, i] = np.mean(dynranges)

    plt.clf()
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label('$\Delta$')
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if logz:
        plt.zscale('log')

    plt.tight_layout()
    plt.savefig(prefix + '_q1_q2_delta_contour.png')
    plt.clf()
    plt.close()

def plot_performance_vs_q(prefix, q, Aq_Asamples_Arates_mc, xlabel, logx=False, logy=False):
    """
    Averaged over all signal inputs
    """

    list_means = []
    list_CIhigh = []
    list_CIlow = []
    for Asamples_Arates_mc in Aq_Asamples_Arates_mc:
        mean, var, std = utilities.bayes_mvs_wrapper(Asamples_Arates_mc.flatten(), alpha=0.95)
        list_means.append(mean[0])
        list_CIhigh.append(mean[1][1] - mean[0])
        list_CIlow.append(mean[0] - mean[1][0])

    plt.errorbar(q, list_means, yerr=[list_CIlow, list_CIhigh], 
        marker='o', ls='-', color='blue')
    # plt.plot(q, list_means, marker='o', ls='-', color='blue')
    plt.ylabel("${MC}$", fontsize=30)
    plt.xlabel(xlabel, fontsize=30)
    plt.ylim(ymin=0.0)
    plt.tight_layout()
    plt.savefig(prefix + "_mc_vs_q.png", dpi=300)
    plt.close()
    plt.clf()

def plot_performance_contour(prefix, q1, q2, listQ1_listQ2_listTrials_mc, 
    xlabel, ylabel, logx=False, logy=False, logz=False, levels=None):

    X, Y = np.meshgrid(q1, q2)
    Z = np.zeros((len(q2),len(q1)))
    for i, listQ2_listTrials_mc in enumerate(listQ1_listQ2_listTrials_mc):
        for j, listTrials_mc in enumerate(listQ2_listTrials_mc):
            Z[j,i] = np.mean(listTrials_mc) # Take average over all signal inputs

    plt.clf()
    if levels==None:
        plt.contourf(X, Y, Z, cmap=cm.magma)
    else:
        plt.contourf(X, Y, Z, cmap=cm.magma, levels=levels)
    cbar = plt.colorbar()
    cbar.set_label('${MC}$')
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

if __name__ == '__main__':
    """
    """

    prefix = 'MC_bin_N500_c1_e10_ws1.132_gain1_p-lin_mu_vs_rsig2'

    # 1D Plot - dynamic range
    final_results = utilities.load_object('/home/nathaniel/workspace/DiscreteESN/dynamic_range_code/' + prefix + '_final_results.pyobj')
    print(final_results['parameters'])
    # structure: Q1 -> Q2 -> samplings -> success_rates -> (q1, q2, p, (MC, delay, coeff), F)
    Aq1_Aq2_Asample_Arates_F = np.array([[[ [ result[4] for result in k ] for k in j] 
        for j in i] for i in final_results['results']])
    Aq1_Aq2_Asample_Arates_p = np.array([[[ [ result[2] for result in k ] for k in j] 
        for j in i] for i in final_results['results']])
    Aq1_Aq2_Asample_Arates_MC = np.array([[[ [ result[3][0] for result in k ] for k in j] 
        for j in i] for i in final_results['results']])

    plot_mc_vs_p(prefix, Aq1_Aq2_Asample_Arates_p[0,0,0,:], Aq1_Aq2_Asample_Arates_MC[0,0,0,:])

    plot_F_vs_p(prefix, Aq1_Aq2_Asample_Arates_p[0,0,0,:], Aq1_Aq2_Asample_Arates_F[0,0,0,:])

    plot_dynrange_vs_q(prefix, final_results['parameters']['q1_list'], 
        Aq1_Aq2_Asample_Arates_p[:,0,:,:], Aq1_Aq2_Asample_Arates_F[:,0,:,:])

    plot_performance_vs_q(prefix, final_results['parameters']['q1_list'], 
        Aq1_Aq2_Asample_Arates_MC[:,0,:,:], "$\mu$", logx=False, logy=False)

    plot_performance_contour(prefix, final_results['parameters']['q1_list'], 
        final_results['parameters']['q2_list'], Aq1_Aq2_Asample_Arates_MC, 
        "$\mu$", "$r_{sig}$", logx=False, logy=False, logz=False, levels=None)

    plot_dynrange_contour(prefix, final_results['parameters']['q1_list'], 
        final_results['parameters']['q2_list'], Aq1_Aq2_Asample_Arates_p, Aq1_Aq2_Asample_Arates_F,
    "$\mu$", "$r_{sig}$", logx=False, logy=False, logz=False, levels=None)
