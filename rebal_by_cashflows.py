import numpy as np
from itertools import permutations
from itertools import compress


p1 = np.matrix([1.0, 1.0, 10.0, 20.0])  # prices in the period before
p2 = np.matrix([1.0, 2.0, 7.0, 20.0])  # new prices
q1 = np.matrix([100.0, 55.55, 20.0, 10.0])  # initial amount owned for each token
no_assets = p1.shape[1]
dollar_in = 100.0
dollar_out = -100.0


def port_val(prc=None, qnty=None):
    return np.matmul(prc, qnty.transpose())[0, 0]


def wght_fnc(prc=None, qnty=None):
    # weights are scaled by 10,000 so that 1 bps = 1
    # anything below 1 bps will cause issues
    return (np.multiply(prc, qnty) / np.matmul(prc, qnty.transpose())[0, 0]) * 10000


def min_fnc(w1_d=None, w2_d=None):
    # w1_d is a given diagonal matrix of previous period weights * 10,000
    # w2_d is a the variable, a diagonal matrix of current period weights * 10,000
    return np.matmul(np.ones(no_assets).transpose(), np.matmul(((w2_d - w1_d) ** 2), np.ones(no_assets)))


def loop1(prc2=None, qnty1=None, dollr_chng=100.0, tgt=None, divisor=50):
    # create diagonal matrix for the target matrix
    tgt_diag = np.zeros((no_assets, no_assets))
    np.fill_diagonal(tgt_diag, tgt)

    # create all permutations (given the step size) and select only permutations where sum = dollar change
    arr = np.arange(0, dollr_chng + dollr_chng / divisor, dollr_chng / divisor)
    perms = list(permutations(arr, no_assets))
    sub_combos = list(compress(perms, np.sum(list(perms), 1) == dollr_chng))

    # sub_combos2 = []
    #for item in perms:
    #    if np.sum(item) == dollr_chng:
    #        sub_combos2.append(item)

    best_min = 9999999999999999
    best_w = None
    best_qnty = None

    # loop through all final permutations to find the 1 that minimizes the squared difference
    # between the final weights and target weights
    for i, cmbo in enumerate(sub_combos):
        temp_qnty = np.divide(sub_combos[i], prc2) + qnty1
        temp_w = wght_fnc(prc2, temp_qnty)
        temp_w_diag = np.zeros((no_assets, no_assets))
        np.fill_diagonal(temp_w_diag, temp_w)
        temp_min = min_fnc(tgt_diag, temp_w_diag)
        if temp_min < best_min:
            best_qnty = temp_qnty
            best_w = temp_w
            best_min = temp_min
    return best_w, best_qnty


# main call
w1 = wght_fnc(p1, q1)
target = w1
w_out_dca_in, q_out_dca_in = loop1(p2, q1, dollar_in, target, 50)
w_out_dca_out, q_out_dca_out = loop1(p2, q1, dollar_out, target, 50)

q_change_dca_in = q_out_dca_in - q1
q_change_dca_out = q_out_dca_out - q1

print('the amount of tokens to buy given the dollar amount coming in is ')
print(q_change_dca_in)
print('the amount of tokens to buy given the dollar amount going out is ')
print(q_change_dca_out)


