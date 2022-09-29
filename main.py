from __future__ import division
import numpy as np
import pandas as pd


# VARIABLES
ann_ret = 0.07
ann_ret_adj = 0.0
ann_std = 0.19
ann_cash_ret = 0.02
ann_cash_std = 0
p = 12  # number of periods per year, 4 for quarter, 12 for months, etc.
a = 30  # number of years
n = 50000  # number of trials
st_val = 100.00
# set amount of periods to dca over
# dca_m_vals = range(6, 37,6)
dca_m = 12  # number of periods to DCA in (if p = 12 and dca_m = 12, DCA in 1/12 per month for first year)


def lognrm_mean_calc(per_ret, per_std):
    ln_nrm_mean = np.log(((1 + per_ret) ** 2) / np.sqrt(per_std ** 2 + ((1 + per_ret) ** 2)))
    return ln_nrm_mean


def lognrm_std_calc(per_ret, per_std):
    ln_nrm_std = np.sqrt(np.log(1 + ((per_std ** 2) / ((1 + per_ret) ** 2))))
    return ln_nrm_std


def dca_vals(st_val, dca_m, m):
    out = np.zeros(m)
    for i, val in enumerate(out):
        if i >= dca_m:
            out[i] = 0
        else:
            out[i] = (st_val / dca_m)
    return out


def lump_vals(m, st_val):
    lump_annuity = np.zeros(m)
    lump_annuity[0] = st_val
    return lump_annuity


def period_data(p, a, ann_ret, ann_ret_adj, ann_std, ann_cash_ret):
    m = p * a  # number of periods
    per_ret = (ann_ret - ann_ret_adj) / p
    per_std = ann_std / np.sqrt(p)
    per_ret_cash = (1 + ann_cash_ret) ** (1 / p) - 1
    # per_ret_cash = ann_cash_ret / p
    return m, per_ret, per_std, per_ret_cash

def cum_fv(annuity, rands, m, st_val, per_ret_cash):
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


def lognormal_rands(mean, std, m):
    lognrm_rands = np.random.lognormal(mean, std, m) - 1
    return lognrm_rands


def mc(n, m, mean, std, dca_annuity, lump_annuity, st_val, per_ret_cash, rand_function):
    # empty arrays for final values
    final_dca_vals = []
    final_lump_vals = []
    # run MC and append to tables
    for i in range(n):
        # THIS CREATES RANDOM RETURNS
        rands = rand_function(mean, std, m)
        # calculate values at each period
        dca_fv = cum_fv(dca_annuity, rands, m, st_val, per_ret_cash)
        lump_fv = cum_fv(lump_annuity, rands, m, st_val, per_ret_cash)
        # return final value
        final_dca_vals.append(dca_fv[m - 1])
        final_lump_vals.append(lump_fv[m - 1])
    return final_dca_vals, final_lump_vals


def summary(final_dca_vals, final_lump_vals):
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
m, per_ret, per_std, per_ret_cash = period_data(p, a, ann_ret, ann_ret_adj, ann_std, ann_cash_ret)
# gets log normal mean and std for risk asset and lump annuity
ln_nrm_mean = lognrm_mean_calc(per_ret, per_std)
ln_nrm_std = lognrm_std_calc(per_ret, per_std)
lump_annuity = lump_vals(m, st_val)
dca_annuity = dca_vals(st_val, dca_m, m)
final_dca_vals, final_lump_vals = mc(n, m, ln_nrm_mean, ln_nrm_std, dca_annuity, lump_annuity, st_val, per_ret_cash,
                                     lognormal_rands)
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