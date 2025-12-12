"""
Utils for mTRF analyses
"""

import numpy as np


def segment(arr, n_segments):
    """
    Cuts the data into equally-sized fragments for mTRFpy. Takes an array and the number of segments 
    Works on arrays of different dimensions but uses slices it along len(arr) i.e. along the first dimension or horizontally
    """
    segment_size = len(arr) // n_segments  # Calculate the size of each segment
    segments = [arr[i * segment_size : (i + 1) * segment_size] for i in range(n_segments)]  # Slice the array into 10 segments
    return segments

def normalize_responses(responses):
    """Function that normalizes the EEG data
    Data is normalized by subtracting the
    global mean and dividing by the global
    standard deviation of the data to preserve
    relative amplitudes between channels
   
    Args:
    responses: list of numpy arrays, each array
    contains the EEG data for one trial. The
    data must be organized as n_samples x n_channels
    """

    # Check dimensions
    if isinstance(responses, list):
        n_rows, n_cols = responses[0].shape
        if n_rows < n_cols:
            raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')
        responses_concatenated = np.concatenate(responses, axis=0).flatten()
    else:
        raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')

    global_mean = np.mean(responses_concatenated)
    global_std = np.std(responses_concatenated)
    responses = [(response - global_mean) / global_std for response in responses]

    return responses


def normalize_stimuli(stimuli, axis = 'all'):
    """Function that normalizes the stimuli data
    Data is normalized by dividing each feature
    by the maximum value of that feature

    Args:
    stimuli: list of numpy arrays, each array
    contains the stimuli data for one trial. The
    data must be organized as n_samples x n_features
    """

    # Check dimensions
    if isinstance(stimuli, list):
        n_rows, n_cols = stimuli[0].shape
        if n_rows < n_cols:
            raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')  
        stimuli_concatenated = np.concatenate(stimuli, axis=0)
    else:
        raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')

    if axis == 'ind':
        feats_max = np.max(stimuli_concatenated, axis=0)
    elif axis == 'all':
        feats_max = np.max(stimuli_concatenated)
    stimuli = [stimulus / feats_max for stimulus in stimuli]

    return stimuli


def shuffle_surprisal(original_stimuli):
    """
    Shuffles surprisal values (but NOT the onset positions) in a stimuli support vector. 
    stimuli: a vector of shape timepoints x features (2 features: onset and stimuli)
        (if working with normalized stimuli arrays, input the index: stimuli_list_normalized[i])
    
    ----
    returns the stimuli array 
    """

    assert isinstance(original_stimuli, np.ndarray)
    assert original_stimuli.shape[1] == 1 or original_stimuli.shape[1] == 2
    
    if original_stimuli.shape[1] == 2:
        stimuli_toshuffle = original_stimuli.copy()[:,1] #if the stimulus has both the onset and surprisal vectors stacked, shuffle the second one
    else:
        stimuli_toshuffle = original_stimuli.copy()
        
    # Find non-zero values and their indices
    nonzero_indices = np.nonzero(stimuli_toshuffle) 
    nonzero_values = stimuli_toshuffle[nonzero_indices]

    np.random.shuffle(nonzero_values)
    stimuli_toshuffle[nonzero_indices] = nonzero_values

    return stimuli_toshuffle


def shuffle_onsets(original_stimuli):
    """ 
    Shuffles the POSITION of the onsets but not their values
    original_stimuli: nd array. 
        Future: might add a way to handle arrays with Onset vector expected to occpy the first row of the array (array[:,0])
    ----
    Returns: an array of 0s and 1s with the same number of 1s as in the original stimulus

    """
    assert isinstance(original_stimuli, np.ndarray)
    assert original_stimuli.shape[1] < 2 #

#ensuring that the dimensions are correct, if it's a support vector with more than one feature such as with onset+suprisal, takes the first feature only
    if original_stimuli.shape[1] == 1:
        shuffled_stimuli = original_stimuli.copy()
        #shuffle positions of onset
        np.random.shuffle(shuffled_stimuli)

#going to ignore how to deal with multiple features for now
    """ elif original_stimuli.shape[1] == 2: 
        shuffled_stimuli = original_stimuli.copy()
        np.random.shuffle(shuffled_stimuli[:,0])
    """

    return shuffled_stimuli