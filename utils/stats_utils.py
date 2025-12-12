import numpy as np
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind, kstest, wilcoxon, pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import pickle
import mne
import pandas as pd

import glob
import os
from update_sub_lists import*
from pp_utils import *


### T-test utils ###

def gaussian_test(array, axis = 0):
    """
    KS test to determine whether the data is normally distributed. 
    Takes an array of shape [n_participants, n_timepoints] and determines whether distribution 
    is normal at each timepoint.
    axis: which axis of the array to test the array over
    ---
    Returns:
    significance: p-value of KS test averaged over all timepoints
    """
    n_points = array.shape[axis]
    p_values = []
    
    for timepoint in range(0, n_points):
        res = kstest(array[:,timepoint], 'norm')
        p_values.append(res.pvalue)

    p_values = np.array(p_values)
    print(f'testing gaussianity over {n_points} points')
    significance = p_values.mean()

    if significance > 0.05: 
        print("Distribution is normal. p = ", significance)
    elif significance < 0.04:
        print("Distribution is not normal. p = ", significance)

    return significance

def t_within_points(array, lim1 = None, lim2 = None):
    """ 
    compare within-subjects samples along an axis of points (e.g. times or freqs)
    uses the 1-sample t-test if normal and wilcoxon otherwise

    array: array of shape n_subs x n_points
    subaxis: which dim contains the subjects
    pointaxis: which dim contains the 
    lim1, lim2: indices of the points over which we want to compute differences (e.g. from 4-30 Hz). 
        Find the indices beforehand using index_custom()
        (prevents loss of power through multiple comparisons) 
    ---
    returns: (both of shape n_points)
    t: a list of t statistics
    p: a listp values, one p value for each time point
    """
    n_points = array.shape[1]
    ks = gaussian_test(array, axis = 0)


    test_stats = []
    p_values = []

    if ks > 0.05:
        print('using 1samp ttest')
        test_function = lambda array, point: ttest_1samp(array[:, point])
    elif ks < 0.05:
        print('using wilcoxon test')
        test_function = lambda array, point: wilcoxon(array[:, point])

    for point in range(n_points):
        res = test_function(array, point)
        test_stats.append(res.statistic)
        p_values.append(res.pvalue)
    

    #FDR CORRECTION
    if lim1 != None:
        assert lim2 != None, 'define both lim1 and lim 2, otherwise both should be None'

        #correct order of lims in case lim2 > lim 1
        upper_lim = np.max([lim1, lim2])
        lower_lim = np.min([lim1, lim2])

        p_values_lim = p_values[lower_lim:upper_lim]
        p_values_corrected = fdrcorrection(p_values_lim)[1]
        p_values_corrected = np.pad(p_values_corrected, (lower_lim,len(p_values)-upper_lim), mode='constant', constant_values=1)


        print(f'fdr correction over {len(p_values_lim)} points')
    else:
        p_values_corrected = fdrcorrection(p_values)[1]
        print('fdr correction over all p values')
        

    return test_stats, p_values_corrected


    
def invert_p_values(p_values_arr):
    """ 
    Transforms an arr of p values to prepare for plotting using mne.viz.plot_topomap
    Inverts the p values (replace x with 1-x) so that smaller p values 
    Makes all insignificant p values 0 so that when inverted, it shows up as white in the topomap

    p_values_arr: a 1d array of p-values
    returns: p_values_inv


    """
    p_values_inv = [1-x if x <0.05 else 0 for x in p_values_arr]
    return p_values_inv

def t_over_channels(array):
    """
    takes a difference array of size n_subs, n_ch (64 expected) and tests whether the differences are sigificantly different from 0

    ---
    returns:
    t: test statistic of size 64, one for each channel
    p: test statistic of size 64
    """
    t_values = []
    p_values = []
    
    for ch in range(64):
        diff_to_test = array[:, ch]
        t, p = wilcoxon_1samp(diff_to_test)
        t_values.append(t)
        p_values.append(p)

    p_values = fdrcorrection(p_values)[1]
    
    return t_values, p_values

def p_mask(diffs, p_values):

    """ 
    masks differences between two conditions so that only significant differences are plotted on the topomap
    diffs: 1d array
    p_values: 1d array of same size
    """
    assert diffs.shape == p_values.shape, 'diffs and p-values should be the same shape'
    diffs_masked = diffs.copy()
    diffs_masked[p_values>0.05] = 0
    return diffs_masked


### BOOTSTRAPPING UTILS ###

def jackknife_1samp(data_point, test):
    """
    estimate distribution of p values (also test statistics)
    data_freq: data of shape n_subs, already indexed for the freq or timepoint of interest
    test: which test to do. Usually wilcoxon

    ---
    returns:
    p_vals: distribution of p values
    ---
    """
    n_subs = data_point.shape[0]

    t_stat_jk = []
    p_val_jk = []
    
    for sub in range(n_subs):
        data_resampled = np.delete(data_point, sub)
        res = test(data_resampled)
        p_val_jk.append(res.pvalue)
        t_stat_jk.append(res.statistic)

    return t_stat_jk, p_val_jk
        

def subset_epochs_custom(epochs_df, n_epochs, ep_col = 'epochs', ave = False):
    """ 
    Takes a subset of each epoch type
    """
    if ave:
        epochs_sub = np.mean([epoch[np.random.choice(epoch.shape[0], n_epochs, replace=True), :, :]
                                        for epoch in epochs_df[ep_col]], axis = 1)
    else:
        epochs_sub =[epoch[np.random.choice(epoch.shape[0], n_epochs, replace=True), :, :]
                                        for epoch in epochs_df[ep_col]]
        
    epochs_df_new = epochs_df.copy()
    epochs_df_new[ep_col] = [epochs_sub[i] for i in range(epochs_df.shape[0])]

    return epochs_df_new




def bootstrap_diffs(epochs_df_1, epochs_df_2, time_idx, n_iter = 200, n_samp = 100, ep_col = 'epochs'):
    """ 
    epochs_df_1, epochs_df_2:epochs to compare.Dataframe with columns 'subject', 'period', 'musician', 'epochtype', 'epochs'
    period: 'pre' or 'post' training
    time_to_plot: timepoint to bootstrap
    n_iter: number of bootstrap iterations
    n_samp: number of epochs from each epoch df to sample for each iteration
    ---
    Returns: distribution of means of the bootstrapped differences between epochs_df_1 and epochs_df_2 at all channels
        shape n_iter x n_channels
    """

    diffs_fo_boot = []

    for i in range(n_iter):

        #take a random sample of epochs from each df
        epochs_df_firsts_sub = subset_epochs_custom(epochs_df_1, n_samp, ep_col)
        epochs_df_others_sub = subset_epochs_custom(epochs_df_2, n_samp, ep_col)

        #find the mean of each sample
        epochs_df_firsts_sub['evokeds'] =  epochs_df_firsts_sub['epochs'].apply(lambda x: np.mean(x, axis=0)) 
        epochs_df_others_sub['evokeds'] = epochs_df_others_sub['epochs'].apply(lambda x: np.mean(x, axis = 0))

        #calculate diff between mean of firsts and mean of others
        df_diff_sub = epochs_df_firsts_sub[['subject', 'period']].copy()
        df_diff_sub['diff'] = epochs_df_firsts_sub['evokeds'] - epochs_df_others_sub['evokeds']
        # df_diff_sub['diff'] = epochs_df_firsts_sub['evokeds'].reset_index(drop=True) - epochs_df_others_sub['evokeds'].reset_index(drop=True)

        #extract diff values for each suject, retains the 64 EEG channels and the timepoint of interest
        diffs_fo_sub = np.stack(df_diff_sub['diff'].values)[:, :64, time_idx].squeeze()
        diffs_fo_boot.append(diffs_fo_sub)


    diffs_fo_boot = np.stack(diffs_fo_boot)
    diffs_fo_boot_mean = np.mean(diffs_fo_boot, axis = 1) #average over subjects over each iteration

    return diffs_fo_boot_mean

def ci95_bs(arr, axis):
    """ 
    CI95 calculation for bootstrapped means 
    arr: array of shape n_iter x n_channels

    axis: which axis to calculate the CI95 over (0 if n_iter is 0th dimension)
    """
    lower_bound = np.percentile(arr, 2.5, axis = axis)
    upper_bound = np.percentile(arr, 97.5, axis = axis)

    return lower_bound, upper_bound