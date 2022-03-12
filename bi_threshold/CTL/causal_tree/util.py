import os
import errno
import numpy as np
from scipy.stats import ttest_ind
import subprocess
import time


def check_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def divide_set(x, y, t, col, value):
    idx1 = x[:, col] >= value
    idx2 = ~idx1

    x1 = x[idx1]
    x2 = x[idx2]

    y1 = y[idx1]
    y2 = y[idx2]

    t1 = t[idx1]
    t2 = t[idx2]

    return x1, x2, y1, y2, t1, t2


def tau_squared(y, t):
    total = y.shape[0]

    return_val = (-np.inf, -np.inf)

    if total == 0:
        return return_val

    treat_vect = t

    effect = ace(y, treat_vect)
    err = (effect ** 2) * total

    return effect
   

def tau_squared_trigger(outcome, treatment, min_size=1, quartile=False):

    total = outcome.shape[0]

    return_val = (-np.inf, -np.inf, -np.inf)

    if total == 0:
        return return_val

    unique_treatment = np.unique(treatment)
   # unique_treatment = np.round_(unique_treatment,decimals= 1)
    unique_treatment = np.round_(unique_treatment,decimals= 1)
    unique_treatment = np.unique(unique_treatment)


    unique_treatment = (unique_treatment[:] + unique_treatment[:]) / 2
    unique_treatment = unique_treatment[1:-1]
    unique_treatment = np.round_(unique_treatment,decimals= 1)
   # unique_treatment = unique_treatment[unique_treatment != 0.0]
   # unique_treatment = unique_treatment[unique_treatment != 1.0]
    unique_treatment = np.unique(unique_treatment)

    
    

    if(unique_treatment.shape[0] == 1)|(unique_treatment.shape[0] == 0):
        return return_val

    if quartile:
        first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
        third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))

        unique_treatment = unique_treatment[first_quartile:third_quartile]

    threshold_up = unique_treatment
    threshold_down = threshold_up[::-1]
    t_d = np.array([i for i in threshold_down for j in threshold_up if i>j])
    t_u = np.array([j for i in threshold_down for j in threshold_up if i>j])
    #mask = np.where(threshold_down>threshold_up,True,False)
    
    #t_u = threshold_up[mask]
    #t_d = threshold_down[mask]
   # print(t_d.shape[0])

    if (t_d.shape[0] in [0,1]) | (t_u.shape[0] in [0,1]):
        return return_val

    yy = np.tile(outcome, (t_d.shape[0], 1))
    tt = np.tile(treatment, (t_d.shape[0], 1))
    x = np.transpose(np.where((np.transpose(tt) > t_u) & (np.transpose(tt) <= t_d), True, False))
    tt[x] = 1
    tt[np.logical_not(x)] = 0
    treat_num = np.sum(tt == 1, axis=1)
    cont_num = np.sum(tt == 0, axis=1)
    min_size_idx = np.where(np.logical_and(treat_num >= min_size, cont_num >= min_size))
    t_d = t_d[min_size_idx]
    t_u = t_u[min_size_idx]
    tt = tt[min_size_idx]
    yy = yy[min_size_idx]
    if tt.shape[0] == 0:
        best_effect,best_split_d,best_split_u = (-np.inf,-np.inf,-np.inf)
        return best_effect,best_split_d,best_split_u
    y_t_m = np.sum((yy * (tt == 1)), axis=1) / np.sum(tt == 1, axis=1)
    y_c_m = np.sum((yy * (tt == 0)), axis=1) / np.sum(tt == 0, axis=1)
    effect = y_t_m - y_c_m
    err = effect ** 2
    max_err = np.argmax(err)
    best_effect = effect[max_err]
    best_err = err[max_err]
    best_split_d=t_d[max_err]
    best_split_u=t_u[max_err]

    best_err = total * best_err

    return best_effect,best_split_d,best_split_u

def ace(y, t):
    treat = t >= 0.5
    # control = t == 0
    control = ~treat

    yt = y[treat]
    yc = y[control]

    mu1 = 0.0
    mu0 = 0.0
    if yt.shape[0] != 0:
        mu1 = np.mean(yt)
    if yc.shape[0] != 0:
        mu0 = np.mean(yc)

    return mu1 - mu0

def ace_trigger(y, t, trigger_d, trigger_u):
    treat = (t <= trigger_d)& (t >= trigger_u)
    control = ~treat

    yt = y[treat]
    yc = y[control]

    mu1 = 0.0
    mu0 = 0.0
    if yt.shape[0] != 0:
        mu1 = np.mean(yt)
    if yc.shape[0] != 0:
        mu0 = np.mean(yc)

    return mu1 - mu0


def get_pval(y, t):
    treat = t == 1
    # control = t == 0
    control = ~treat

    outcome_cont = y[treat]
    outcome_trt = y[control]

    p_val = ttest_ind(outcome_cont, outcome_trt)[1]

    if np.isnan(p_val):
        return 0.000

    return p_val

def get_pval_trigger(y, t, trigger_d, trigger_u):
    treat = (t <= trigger_d) & (t >= trigger_u)
    control = ~treat

    outcome_cont = y[treat]
    outcome_trt = y[control]

    p_val = ttest_ind(outcome_cont, outcome_trt)[1]

    if np.isnan(p_val):
        return 0.000

    return p_val

def min_size_value_bool(min_size, t, trigger=0.5):
    nt, nc = get_treat_size(t, trigger=trigger)

    return nt, nc, nt < min_size or nc < min_size


def check_min_size(min_size, t, trigger=0.5):
    nt, nc = get_treat_size(t, trigger)

    return nt < min_size or nc < min_size


def get_treat_size(t, trigger=0.5):
    treated = t >= trigger
    control = ~treated
    num_treatment = t[treated].shape[0]
    num_control = t[control].shape[0]

    return num_treatment, num_control


def variance(y, t):
    treat_vect = t

    treat = treat_vect == 1
    # control = treat_vect == 0
    control = ~treat

    if y.shape[0] == 0:
        return np.array([np.inf, np.inf])

    yt = y[treat]
    yc = y[control]

    if yt.shape[0] == 0:
        var_t = np.var(y)
    else:
        var_t = np.var(yt)

    if yc.shape[0] == 0:
        var_c = np.var(y)
    else:
        var_c = np.var(yc)

    return var_t, var_c


def variance_trigger(y, t, trigger):
    treat_vect = t

    treat = treat_vect >= trigger
    # control = treat_vect == 0
    control = ~treat

    if y.shape[0] == 0:
        return np.array([np.inf, np.inf])

    yt = y[treat]
    yc = y[control]

    if yt.shape[0] == 0:
        var_t = np.var(y)
    else:
        var_t = np.var(yt)

    if yc.shape[0] == 0:
        var_c = np.var(y)
    else:
        var_c = np.var(yc)

    return var_t, var_c


def col_dict(names):
    feat_names = {}
    for i, name in enumerate(names):
        column = "Column %s" % i
        feat_names[column] = name
    return feat_names
