import numpy as np
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind, kstest, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import pickle
import mne
import pandas as pd
import matplotlib.pyplot as plt

import glob
import os
from update_sub_lists import*
from pp_utils import *


def find_existing_subjects(data_dir):
    """ 
    automatically find subjects existing in the evoked or epochs folder
    NEED TO BETTER DEFINE HOW TO HANDLE PERIODS, for now it's not relevant
    (subjects are expected to have data in both pre and post periods)
    ---
    returns:
    subjects_to_process: list of subjects
    """
    
    subjects_to_process = []
    for file_name in os.listdir(data_dir):
        # Check if the file matches the format you're interested in

            # Extract the subject ID (last two characters before file extension name)
            subject_id = file_name.split("_")[-1].split(".")[0]
            subjects_to_process.append(subject_id)

    # Sort and filter the list of subjects
    subjects_to_process = sorted(set(subjects_to_process))
    subjects_to_process = [x for x in subjects_to_process if x.isdigit()] 
    return subjects_to_process

# def load_erp_times():
#     """ 
#     avoid because it only works for erps from -0.2 to 0.5
#     """
#     with open('../utils/erp_times.pkl', 'rb') as file:
#         times = pickle.load(file)
#         return times
    
def create_erp_times(tmin, tmax, fs):
    """ 
    creates a time vector for ERPs given start and end time. result should match the .times attribute in epochs
    tmin, tmax: times in seconds
    fs: sampling frequency
    ---
    returns 1d vector with the time
    """
    step = 1 / fs
    times = np.arange(tmin, tmax+step, step)
    return times

def load_channels(ch_file = '../utils/ch_names.pkl'):
    #get channel names
    with open(ch_file, 'rb') as file:
        ch_names_all = pickle.load(file)

    ch_names_64 = ch_names_all[0:64]
    ch_names_72 = ch_names_all[0:72]
    

    return ch_names_64, ch_names_72

def load_ep_info():
    with open('../utils/epochs_info.pkl', 'rb')as file:
        ep_info = pickle.load(file)
    return ep_info



def ch_index(ch_list): 
    """
    finds indices of channels given the channel names
    ch_list: list of channel names, e.g. ['FCz', 'Cz']
    ---
    returns: list of channel indices
    """
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, 'ch_names.pkl')  # Assuming the pkl file is in the same folder as this script
    
    assert isinstance(ch_list, list)
    with open(pkl_path, 'rb') as file:
        ch_names_all = pickle.load(file)

    ch_names_72 = ch_names_all[0:72]
    ch_indices = [ch_names_72.index(item) if item in ch_names_72 else None for item in ch_list]
    return ch_indices

# def time_index(timepoints):
    

#     """
#     Finds the index in the time vector from -0.2 to 0.5
#     timepoints: a list of timepoints expressed in seconds
#     erp_times: time vector to find the index for(which should be an array)
    
#     ---
#     Returns a list of indices  
#     """
#     erp_times = load_erp_times()
#     assert isinstance(timepoints, list)
#     idx_list = []
#     for time in timepoints: 
#         time_idx = min(range(len(erp_times)), key=lambda i: abs(erp_times[i] - time))
#         idx_list.append(time_idx)
#     return idx_list

def time_index_custom(timepoints, erp_times):

    """
    Finds the index in the time vector 
    timepoints: a list of timepoints expressed in seconds
    erp_times: time vector to find the index for(which should be an array)
    
    ---
    Returns a list of indices  
    """
    assert isinstance(timepoints, list)
    idx_list = []
    for time in timepoints: 
        time_idx = min(range(len(erp_times)), key=lambda i: abs(erp_times[i] - time))
        idx_list.append(time_idx)
    return idx_list

def index_custom(timepoints, erp_times):
### THIS IS IDENTICAL TO TIME_INDEX_CUSTOM... with a name that is more clear
    """

    Finds the index in the time vector 
    timepoints: a list of timepoints expressed in seconds
    erp_times: time vector to find the index for(which should be an array)
    
    ---
    Returns a list of indices  
    """
    assert isinstance(timepoints, list)
    idx_list = []
    for time in timepoints: 
        time_idx = min(range(len(erp_times)), key=lambda i: abs(erp_times[i] - time))
        idx_list.append(time_idx)
    return idx_list


def p_times(arrays_to_compare, channels = 'all', test = wilcoxon, fdr = False, fdr_idx = None):
    """ 
    calculate p values of differences between two arrays with erps

    arrays_to_compare: a list of 2 arrays to compare. For example [test_pre, test_post]. 
        Each array should be shape n_subs x n_channels x n_timepoints

    test: which stat test to use. t-test_ind, wilcoxon, etc.
    fdr: whether to apply fdr correction to the p values
    fdr_range: list, min and max time in seconds to consider for fdr correction. 
        p-value for timepoints outside range will be set to 1 
    ---
    returns: a list of p values, one p value for each time point
    
    """
    p_values = []

    assert arrays_to_compare[0].shape == arrays_to_compare[1].shape, 'arrays to compare should be the same shape'
    assert arrays_to_compare[0].shape[1] == 64 or arrays_to_compare[0].shape[1], 'arrays to compare should have 64 channels'
    
    
    if channels == 'all':
        print('Calculating p-value over mean of all channels')
        array1 = arrays_to_compare[0][:, 0:64]
        array2 = arrays_to_compare[1][:, 0:64]
    elif type(channels) == list:
        print(f'Calculating p-value over {channels}')
        p_ch_idx = ch_index(channels)
        array1 = arrays_to_compare[0][:, p_ch_idx]
        array2 = arrays_to_compare[1][:, p_ch_idx]

        
    else:
        print('Channels should be provided in a list. No p-values computed.')
        exit

    print('evokeds array shape', array1.shape)
    for timepoint in range(0, arrays_to_compare[0].shape[2]):
        res = test(array1.mean(axis = 1)[:, timepoint], array2.mean(axis = 1)[:, timepoint])
        p_values.append(res.pvalue)
    
    if fdr:
        if fdr_idx is None:
            p_values = fdrcorrection(p_values)[1]
        else: 
            #make sure fdr idx is within time range
            assert all(0 <= idx < len(p_values) for idx in fdr_idx), 'fdr_idx out of range of timepoints'

            p_values_range = p_values[fdr_idx[0]:fdr_idx[1]]
            p_values_range = fdrcorrection(p_values_range)[1]
            p_values = np.ones_like(np.array(p_values))
            p_values[fdr_idx[0]:fdr_idx[1]] = p_values_range

    return p_values


# def p_times(arrays_to_compare, channels = 'all', fdr = False): #old version
#     #### DOESN'T HAVE FDR CORRECTION, IMPLEMENT IT BEFORE USING THIS FUNCTION AGAIN
#     """ 
#     calculate p values of differences between the pre- and post-training ERPs
#     currently using ind samples t-test but should reconsider this...

#     arrays_to_compare: a list of 2 arrays to compare. Like [test_pre, test_post]
#     returns: a list of p values, one p value for each time point
#     """
#     p_values = []
#     if channels == 'all':
#         print('Calculating p-value over mean of all channels')
#         for timepoint in range(0, arrays_to_compare[0].shape[2]):
            
#             array1 = arrays_to_compare[0][:, 0:64]
#             array2 = arrays_to_compare[1][:, 0:64]
#             res = ttest_ind(array1.mean(axis = 1)[:, timepoint], array2.mean(axis = 1)[:, timepoint])
#             p_values.append(res.pvalue)

    
#     elif type(channels) == list:
#         print(f'Calculating p-value over {channels}')
#         for timepoint in range(0, arrays_to_compare[0].shape[2]):
#             p_ch_idx = ch_index(channels)
#             array1 = arrays_to_compare[0][:, p_ch_idx]
#             array2 = arrays_to_compare[1][:, p_ch_idx]
#             res = ttest_ind(array1.mean(axis = 1)[:, timepoint], array2.mean(axis = 1)[:, timepoint])
#             p_values.append(res.pvalue)

#     else:
#         print('Valid channel arguments: type list')
#         exit

#     return p_values


#GAUSSIAN TEST FUNCTION HERE IS OLD. SEE STATS UTILS FOR UPDATED VERSION
#def gaussian_test(array, axis = 1):
    """
    KS test to determine whether the data is normally distributed. 
    Takes an array of shape [n_participants, n_timepoints] and determines whether distribution 
    is normal at each timepoint.
    Returns significance level of KS test averaged over all timepoints
    """
"""    n_points = array.shape[axis]
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

    return significance"""

#SEE STATS UTILS
"""
Adaptation of scipy implmentation but comparing one sample with an expected mean of 0
Only one input array needed
returns the result of the wilcoxon test
"""
def wilcoxon_1samp(array):
    pop_mean = np.zeros_like(array)
    res = wilcoxon(array, pop_mean)
    return res


def p_times_1sample(array, channels = 'all', tmin = None, tmax = None):
    """ 
    calculate p values of differences between the pre- and post-training ERPs
    uses the 1-sample t-test if normal and wilcoxon otherwise

    array: preprocessed array containing the average ERP (post minus pre) for each subject
    ---
    returns: a list of p values, one p value for each time point
    """
    if channels == 'all':
        print('Calculating p-value over mean of all channels')
        array_ch_mean = array.mean(axis = 1)
    elif type(channels) == list:
        print(f'Calculating p-value over {channels}')
        p_ch_idx = ch_index(channels)
        array = array[:, p_ch_idx]
        array_ch_mean = array.mean(axis = 1)
    else:
        print('Valid channel arguments: type list')
        exit

    #test normality; do 1-sample t-test if normal, wilcoxon otherwise
    ks = gaussian_test(array_ch_mean)

    p_values = []
    for timepoint in range(0, array.shape[2]):
        if ks > 0.05:
            res = ttest_1samp(array_ch_mean[:, timepoint], popmean = 0)       
        elif ks < 0.05:
            res = wilcoxon_1samp(array_ch_mean[:, timepoint])
        p_values.append(res.pvalue)
        n_tests = len(p_values)
    
    if tmin is not None and tmax is not None:
        #FDR correction for multiple comparisons
        start_idx = time_index([tmin])[0]
        end_idx = time_index([tmax])[0]
        end_padding = n_tests-end_idx
        p_values = fdrcorrection(p_values[start_idx:end_idx])[1]

        #pad the array so that it's the right size but everything outside range of interest is 1 so not significant
        p_values = np.pad(p_values,(start_idx, end_padding), constant_values=1)

    else: 
        p_values = fdrcorrection(p_values)[1] #does fdr correction over all timepoints
        #recommend selecting the time slice of interest, otherwise it reduces the power
        

    return p_values


"""
This function is used to process the data so that it's ready for the 1 sample t-test. Takes the 
subject averages for each condition, and subtracts them
Returns an array of evoked data where the first dim is the number of subjects (supposedly)
"""
def find_diff_sa(evoked_list1, evoked_list2):
    diff_evoked_list = [evoked1 - evoked2 for evoked1, evoked2 in zip(evoked_list1, evoked_list2)]
    diff_evoked_sa = np.stack(diff_evoked_list)
    return diff_evoked_sa




def p_chs(arrays_to_compare, time_idx, ttest):
    """ 
    calculate p values of differences at each channel between the pre- and post-training ERPs
    arrays_to_compare: a list of 2 arrays to compare, e.g. [test_pre, test_post]
    time_idx: timepoint at which you want to compare. Find it from the timepoint in seconds using time_index()
    ttest: 'ind' or 'rel'

    ----
    returns: a list of p values one p value for each channel
    """
    p_values = []
 
    for channel in range(0, 64): #assumes 64 channel eeg
        array1 = arrays_to_compare[0][:, channel, time_idx]
        array2 = arrays_to_compare[1][:, channel, time_idx]

        if ttest == 'ind':
            res = ttest_ind(array1, array2)
        elif ttest == 'rel':
            res = ttest_rel(array1, array2)
        p_values.append(res.pvalue)
    
    #FDR correction 
    p_values = fdrcorrection(p_values)[1]

    return p_values
""" 
Process the p-values so that when they're plotted as a topomap, the small values (i.e. the most significant) are plotted as red
Also anything > 0.05 becomes 0 
"""

def scale_p_channels(p_values, threshold = 0.95):
    scaled_values = [1-x for x in p_values]
    scaled_values = [0 if x < threshold else x for x in scaled_values]


    return scaled_values

def compute_power(epochs, tmin = 0, tmax = 0.25, bands=['delta', 'theta', 'alpha', 'beta', 'gamma', 'all'], method = 'welch'):
    """
    Returns a DataFrame with power computed over each frequency band for given epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs for which to compute the PSD and power.
    bands : list of str, optional
        List of frequency bands to compute power for. Default is ['delta', 'alpha'].
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame where each column represents the power in a different frequency band.
    """

    freqbands = {'delta': [0.5, 4], 
                 'theta': [4, 8],
                 'alpha': [8, 12],
                 'beta': [12, 30],
                 'gamma': [30, 45],
                 'all': [0.5, 45]
                }

    
    power_dict = {}
    for key in bands:
        if key not in freqbands:
            continue  
        fmin, fmax = freqbands[key]

        psd = mne.Epochs.compute_psd(epochs, 
                                     method = method,
                                     fmin=fmin, 
                                     fmax=fmax, 
                                     tmin = tmin, 
                                     tmax = tmax)
        
        psd_ave_64 = psd.average() #average over epochs
        psd_ave = np.mean(psd_ave_64.get_data(), axis = 0) #average over channel

        #integrate PSD
        power = np.trapz(psd_ave)

        # save PSD
        power_dict[key] = power


    df = pd.DataFrame([power_dict])

    return df


def load_evoked_epochs(subjects_to_process, task):

    evoked_dir = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_{task}/{task}_ERP_data'
    epochs_dir = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_{task}/{task}_epochs_data'

    """
    Loads the epochs and evoked .fif files and organizes them into lists to use for plotting and analysis
    subjects_to_process: list of subjects where each element is a string. e.g. ['01', '02']

    ---
    Returns concatenated epochs and evoked lists: concat_epochs_pre, concat_evoked_pre, concat_epochs_post, concat_evoked_post
    """
    evoked_list_pre = []
    epochs_list_pre = []
    evoked_list_post = []
    epochs_list_post = []

    #subject averages
    #epochs_list_pre_sa =[]

    #for file in sorted(os.listdir(evoked_dir)):
    assert isinstance (subjects_to_process, list)
    for subject in subjects_to_process:
        print('Processing subject: ', subject)

        file_evokeds_pre = glob.glob(os.path.join(evoked_dir, f'{task}_ERP_pre_{subject}.fif'))[0]
        file_epochs_pre = glob.glob(os.path.join(epochs_dir, f'{task}_epochs_pre_{subject}.fif'))[0]
    
        evoked_pre = mne.read_evokeds(file_evokeds_pre)[0]
        evoked_list_pre.append(evoked_pre)
        epochs_pre = mne.read_epochs(file_epochs_pre)
        epochs_list_pre.append(epochs_pre)

        file_evokeds_post = glob.glob(os.path.join(evoked_dir, f'{task}_ERP_post_{subject}.fif'))[0]
        file_epochs_post = glob.glob(os.path.join(epochs_dir, f'{task}_epochs_post_{subject}.fif'))[0]
    
        evoked_post = mne.read_evokeds(file_evokeds_post)[0]
        evoked_list_post.append(evoked_post)
        epochs_post = mne.read_epochs(file_epochs_post)
        epochs_list_post.append(epochs_post)


    concat_epochs_pre = mne.concatenate_epochs(epochs_list_pre)
    concat_evoked_pre = mne.combine_evoked(evoked_list_pre, weights = 'equal')

    concat_epochs_post = mne.concatenate_epochs(epochs_list_post)
    concat_evoked_post = mne.combine_evoked(evoked_list_post, weights = 'equal')
    
    return (concat_epochs_pre, concat_evoked_pre, concat_epochs_post, concat_evoked_post)

def load_epochs_bysubject(subjects_to_process, task, epochs_dir, sub_ave = True):


    """
    Loads the epochs and evoked .fif files and organizes them into a dataframe to use for plotting and analysis
    subjects_to_process: list of subjects where each element is a string. e.g. ['01', '02']


    ---
    Returns a dataframe with columns 'subject', 'period', 'musician', and 'epochs'.
        each row of ['epochs'] is an array of shape n_channels x n_timepoints, and is the average of all epochs from one subject
    """
    
    #epochs_dir = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_{task}/{task}_epochs_data'
    epochs_df = pd.DataFrame(columns = ['subject', 'period', 'musician', 'epochs'])
    
    good_listen_subjects, good_motor_subjects, musicians, nonmusicians = load_subject_lists()

    #subject averages
    #epochs_list_pre_sa =[]

    #for file in sorted(os.listdir(evoked_dir)):
    assert isinstance (subjects_to_process, list)

    for subject in subjects_to_process:
        print('Processing subject: ', subject)

        if subject in musicians: 
            musician = 1
        else: 
            musician = 0
        for period in ['pre', 'post']:
            file_epochs_pre = glob.glob(os.path.join(epochs_dir, f'{task}_epochs_{period}_{subject}.fif'))[0]
            epochs_sub = mne.read_epochs(file_epochs_pre)

            if sub_ave:
                epochs_sub = np.mean(epochs_sub.get_data()[:, :64, :], axis = 0) #get only the eeg channels and average all trials per subject
            else: 
                epochs_sub = epochs_sub.get_data()[:,:64, :]
            

            df_sub = pd.DataFrame({
                'subject': subject,
                'period' : period,
                'musician' : musician,
                'epochs': [epochs_sub]
            })
            epochs_df = pd.concat([epochs_df, df_sub])


    epochs_df.reset_index(drop=True, inplace=True)
    return (epochs_df)

def load_error_epochs_bysubject(epochs_dir, subjects_to_process, epoch_type, sub_ave = True):
    """ 
    Loads the epochs for error trials
    subjects_to_process: list of subjects where each element is a string. e.g. ['01', '02']
    epoch_type: which error keystrokes are included. Currently 'all', 'inv', 'shinv' and 'norm'
        ---future: separate keystrokes that are the first keystroke after a map change from all the other keystrokes
    ---
    Returns a dataframe with columns 'subject', 'period', 'musician', and 'epoch_type'.
        each row of ['epochs'] is an array of shape n_channels x n_timepoints, and is the average of all epochs from one subject
    
    """

    epochs_df = pd.DataFrame(columns = ['subject', 'period', 'musician', 'epochtype', 'epochs'])
    good_listen_subjects, good_motor_subjects, musicians, nonmusicians = load_subject_lists()

    assert isinstance (subjects_to_process, list)

    for subject in subjects_to_process:
        print('Processing subject: ', subject)

        if subject in musicians: 
            musician = 1
        else: 
            musician = 0
        for period in ['pre', 'post']:
            file_epochs_pre = glob.glob(os.path.join(epochs_dir, f'error_epochs_{epoch_type}_{period}_{subject}.fif'))[0]
            epochs_sub = mne.read_epochs(file_epochs_pre)

            if sub_ave:
                epochs_sub = np.mean(epochs_sub.get_data()[:, :64, :], axis = 0) #get only the eeg channels and average all trials per subject
            else: 
                #epochs_sub = epochs_sub.get_data()[:, :64, :]
                epochs_sub = epochs_sub.get_data()

            df_sub = pd.DataFrame({
                'subject': subject,
                'period' : period,
                'musician' : musician,
                'epochtype': epoch_type,
                'epochs': [epochs_sub]
            })
            epochs_df = pd.concat([epochs_df, df_sub])


    epochs_df.reset_index(drop=True, inplace=True)
    return (epochs_df)


# def load_error_epochs_bysubject(subjects_to_process, epoch_type, epochs_dir, sub_ave = True):
#     """ 
#     Loads the epochs for error trials
#     subjects_to_process: list of subjects where each element is a string. e.g. ['01', '02']
#     epoch_type: which error keystrokes are included. Currently 'all', 'inv', 'shinv' and 'norm'
#         ---future: separate keystrokes that are the first keystroke after a map change from all the other keystrokes
#     ---
#     Returns a dataframe with columns 'subject', 'period', 'musician', and 'epoch_type'.
#         each row of ['epochs'] is an array of shape n_channels x n_timepoints, and is the average of all epochs from one subject
    
#     """

#     epochs_df = pd.DataFrame(columns = ['subject', 'period', 'musician', 'epochtype', 'epochs'])
#     good_listen_subjects, good_motor_subjects, musicians, nonmusicians = load_subject_lists()

#     assert isinstance (subjects_to_process, list)

#     for subject in subjects_to_process:
#         print('Processing subject: ', subject)

#         if subject in musicians: 
#             musician = 1
#         else: 
#             musician = 0
#         for period in ['pre', 'post']:
#             file_epochs_pre = glob.glob(os.path.join(epochs_dir, f'error_epochs_{epoch_type}_{period}_{subject}.fif'))[0]
#             epochs_sub = mne.read_epochs(file_epochs_pre)

#             if sub_ave:
#                 epochs_sub = np.mean(epochs_sub.get_data()[:, :64, :], axis = 0) #get only the eeg channels and average all trials per subject
#             #epochs_sub = epochs_sub[np.newaxis, :, :]
#             else: 
#                 epochs_sub = epochs_sub.get_data()[:,:64, :]

#             df_sub = pd.DataFrame({
#                 'subject': subject,
#                 'period' : period,
#                 'musician' : musician,
#                 'epochtype': epoch_type,
#                 'epochs': [epochs_sub]
#             })
#             epochs_df = pd.concat([epochs_df, df_sub])


#     epochs_df.reset_index(drop=True, inplace=True)
#     return (epochs_df)


def plot_topo_custom(topo_data, pos, colorbar=False, cbar_label = None, title = None, **kwargs):
    """  
    uses mne plot topo function to plot a nice topomap with my favourite parameters 
    NOT COMPLETELY POLISHED BUT USABLE
    """
    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(topo_data, pos, axes=ax, show=False, **kwargs)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label(label = cbar_label, rotation=270, labelpad=15) 

    if title != None:
        ax.set_title(title)
    plt.show()