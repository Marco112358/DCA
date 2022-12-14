from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as sc


# VARIABLES
ann_ret = 0.15
ann_ret_adj = 0.0
ann_std = 0.5
ann_cash_ret = 0.02
ann_cash_std = 0.0
p = 365  # number of periods per year, 4 for quarter, 12 for months, 365 for days etc.
m = 180  # periods total (so if p = 365 and m = 180, total days = 180)
n = 1000  # number of trials
st_val = 100.0
# set amount of periods to dca over
# dca_m_vals = range(6, 37,6)
dca_m = 30  # number of periods to DCA in (if p = 12 and dca_m = 12, DCA in 1/12 per month for first year)
trans_matrix = np.asmatrix([[0.5901981, 0.4526559], [0.409819, 0.5473441]])
 # [[uu, du],[ud, dd]]


def dca_vals(st_val=100.0, dca_m=30, m=365):
    out = np.zeros(m)
    for i, val in enumerate(out):
        if i >= dca_m:
            out[i] = 0
        else:
            out[i] = (st_val / dca_m)
    return out


def lump_vals(m=365, st_val=100.0):
    lump_annuity = np.zeros(m)
    lump_annuity[0] = st_val
    return lump_annuity


def cum_fv(annuity=None, rands=None, m=365, st_val=100.0, per_ret_cash=0.02):
    # Cash compounds per period... excess interest over starting value stays in cash account....
    final = np.zeros(m)
    out1 = np.zeros(m)
    out2 = np.zeros(m)
    for i, item in enumerate(final):
        if i == 0:
            out1[i] = (annuity[i] * (1 + rands[i]))
            out2[i] = ((st_val - annuity[i]) * (1 + per_ret_cash))
        else:
            out1[i] = (annuity[i] * (1 + rands[i])) + (out1[i - 1] * (1 + rands[i]))
            out2[i] = ((out2[i - 1] - (annuity[i])) * (1 + per_ret_cash))
    final = out1 + out2
    return final


def stdnorm_rndm_var():
    # standard normal variable that should change every call
    x = sc.norm.rvs()
    return x


def prc_new_with_transprobs(prc_prev=0.0, mean=0.0, std=0.0, step=1, days=1, p=365, price_prev2=0.0,
                            transmatrix=np.matrix([[0.0, 0.0], [0.0, 0.0]])):
    # Standard Normal Distribution used
    # Create a new wealth based on Brownian Motion off of Previous Wealth
    mean_fin = mean * days / p
    std_fin = std / np.sqrt(p / days)
    if prc_prev < price_prev2:
        price_new = transmatrix[0, 1] * (prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step +
                                                           std_fin * (step ** 2) * abs(stdnorm_rndm_var()))) + \
                    transmatrix[1, 1] * (prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step +
                                                           std_fin * (step ** 2) * -abs(stdnorm_rndm_var())))
    elif prc_prev == price_prev2:
        price_new = (prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step +
                                                           std_fin * (step ** 2) * stdnorm_rndm_var()))
    else:
        price_new = transmatrix[0, 0] * (prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step +
                                                           std_fin * (step ** 2) * abs(stdnorm_rndm_var()))) + \
                    transmatrix[1, 0] * (prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step +
                                                           std_fin * (step ** 2) * -abs(stdnorm_rndm_var())))
#    elif dist == "log":
#        price_new = prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step + std_fin * (step ** 2) *
#                                      (lognorm_rndm_var(mean, std) - 1))
    return price_new


def rnd_prc_tbl(prc_st=None, mean=0.0, std=0.0, days=1, no_trials=1000, no_periods=1, p=365,
                transmatrix=np.matrix([[0.0, 0.0], [0.0, 0.0]])):
    fin = pd.DataFrame(index=np.arange(0, no_trials), columns=np.arange(0, int(no_periods / days) + 1))
    for n in np.arange(0, no_trials):
        fin.iloc[n, 0] = prc_st
        for t in np.arange(1, int(no_periods / days) + 1):
            if t == 1 or t == 2:
                fin.iloc[n, t] = prc_new_with_transprobs(prc_st, mean, std, 1, days, p, prc_st, transmatrix)
            else:
                fin.iloc[n, t] = prc_new_with_transprobs(fin.iloc[n, t - 1], mean, std, 1, days,  p,
                                                         fin.iloc[n, t - 2], transmatrix)
    return fin


def return_tbl(tbl=None):
    out = pd.DataFrame(index=tbl.index, columns=tbl.columns)
    for i, col in enumerate(tbl):
        if i == 0:
            out.loc[:,0] = 0.0
        else:
            out.loc[:, i] = tbl.loc[:, i] / tbl.loc[:, i-1] - 1
    return out


def mc2(n=1000, m=365, dca_annuity=None, lump_annuity=None, st_val=100.0, per_ret_cash=0.02, rand_rtns=None):
    # empty arrays for final values
    final_dca_vals = []
    final_lump_vals = []
    # run MC and append to tables
    for tr in range(n):
        # THIS CREATES RANDOM RETURNS
        rands = rand_rtns.loc[tr, :]
        # calculate values at each period
        dca_fv = cum_fv(dca_annuity, rands, m, st_val, per_ret_cash)
        lump_fv = cum_fv(lump_annuity, rands, m, st_val, per_ret_cash)
        # return final value
        final_dca_vals.append(dca_fv[m - 1])
        final_lump_vals.append(lump_fv[m - 1])
    return final_dca_vals, final_lump_vals


def summary(final_dca_vals=None, final_lump_vals=None):
    sum_tbl = pd.DataFrame(index=['mean', 'median', 'range', '1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%',
                                  '99%'], columns=['DCA', 'Lump'])
    sum_tbl['DCA']['mean'] = np.mean(final_dca_vals)
    sum_tbl['DCA']['median'] = np.median(final_dca_vals)
    sum_tbl['DCA']['range'] = np.max(final_dca_vals) - np.min(final_dca_vals)
    sum_tbl['DCA']['1%'] = np.percentile(final_dca_vals, 1)
    sum_tbl['DCA']['5%'] = np.percentile(final_dca_vals, 5)
    sum_tbl['DCA']['10%'] = np.percentile(final_dca_vals, 10)
    sum_tbl['DCA']['25%'] = np.percentile(final_dca_vals, 25)
    sum_tbl['DCA']['50%'] = np.percentile(final_dca_vals, 50)
    sum_tbl['DCA']['75%'] = np.percentile(final_dca_vals, 75)
    sum_tbl['DCA']['90%'] = np.percentile(final_dca_vals, 90)
    sum_tbl['DCA']['95%'] = np.percentile(final_dca_vals, 95)
    sum_tbl['DCA']['99%'] = np.percentile(final_dca_vals, 99)

    sum_tbl['Lump']['mean'] = np.mean(final_lump_vals)
    sum_tbl['Lump']['median'] = np.median(final_lump_vals)
    sum_tbl['Lump']['range'] = np.max(final_lump_vals) - np.min(final_lump_vals)
    sum_tbl['Lump']['1%'] = np.percentile(final_lump_vals, 1)
    sum_tbl['Lump']['5%'] = np.percentile(final_lump_vals, 5)
    sum_tbl['Lump']['10%'] = np.percentile(final_lump_vals, 10)
    sum_tbl['Lump']['25%'] = np.percentile(final_lump_vals, 25)
    sum_tbl['Lump']['50%'] = np.percentile(final_lump_vals, 50)
    sum_tbl['Lump']['75%'] = np.percentile(final_lump_vals, 75)
    sum_tbl['Lump']['90%'] = np.percentile(final_lump_vals, 90)
    sum_tbl['Lump']['95%'] = np.percentile(final_lump_vals, 95)
    sum_tbl['Lump']['99%'] = np.percentile(final_lump_vals, 99)

    sum_tbl = sum_tbl.loc[['mean', 'median', 'range', '1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%']]
    return sum_tbl


# Calcs returns and std per period
per_ret_cash = (1 + ann_cash_ret) ** (1 / p) - 1
# gets log normal mean and std for risk asset and lump annuity
lump_annuity = lump_vals(m, st_val)
dca_annuity = dca_vals(st_val, dca_m, m)

rnd_prcs = rnd_prc_tbl(st_val, ann_ret, ann_std, 1, n, m,  p, trans_matrix)
rand_rtns = return_tbl(rnd_prcs)

final_dca_vals, final_lump_vals = mc2(n, m, dca_annuity, lump_annuity, st_val, per_ret_cash, rand_rtns)

summary_table = summary(final_dca_vals, final_lump_vals)
print(summary_table)

# if you want to test different DCA initial lengths, you can do that below

#for i, val in enumerate(dca_m_vals):
#    # set dca period and get dca annuity
#    dca_m = dca_m_vals[i]
#    dca_annuity = dca_vals(st_val, dca_m, m)
#    final_dca_vals, final_lump_vals = mc(n, m, ln_nrm_mean, ln_nrm_std, dca_annuity, lump_annuity, st_val, per_ret_cash, lognormal_rands)
#    summary_table = summary(final_dca_vals, final_lump_vals)
#    print(summary_table)