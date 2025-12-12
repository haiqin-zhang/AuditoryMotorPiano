"""
Functions for preprocessing
"""

import numpy as np
import mne

#======================================================================================
#                      PROCESSING TRIGGERS
#======================================================================================
"""
Filters the trigger signals so that only the first value of each group of triggers that is close in time is retained. 
Necessary because the triggers are analog so each real trigger results in a string of triggers being detected.

Trig array: the array straight out of mne.find_events
Threshold: the minimum difference in the first column between the previous and current row for the current row to be retained 
""" 
def clean_triggers(trig_array, threshold=100):
    if not np.any(trig_array):
        return trig_array
    else:
        cleaned_triggers = []
        prev_trigger_time = trig_array[0, 0]  # Initialize with the first trigger time
        cleaned_triggers.append(trig_array[0])  # Retain the first trigger
        
        for trigger in trig_array[1:]:
            trigger_time = trigger[0]
            if trigger_time - prev_trigger_time > threshold:
                cleaned_triggers.append(trigger)
                prev_trigger_time = trigger_time
        
        return np.array(cleaned_triggers)

"Preserves short events during downsampling"
def discretize(arr, final_length, downfreq_factor = 32):
    n_bins = int(len(arr)//downfreq_factor)

    if len(arr) % downfreq_factor != 0:
        remainder = downfreq_factor - (len(arr) % downfreq_factor)
        arr = np.append(arr, [0] * int(remainder))
        n_bins +=1

    arr_reshaped = arr.reshape(n_bins, downfreq_factor)
    discretized_arr = np.any(arr_reshaped, axis=1).astype(int)

    discretized_arr = discretized_arr[0:final_length]

    return discretized_arr


"""Takes a support vector (0 and 1s with 1s being events at time index) and makes 
an event erray compatible with mne.raw 

Returns: events_arr, nevents x 3 array where 1st column is event timestamp (relative to 
beginning of trial), 2nd column is zeros because we don't use it, 3rd column is event type 
from 2 to 6 (corresponding to the trigger channel numbers)
"""
def make_raw_events(supportvec):
    events_arr = []
    
    for i in range(2, 7):
        indices = np.where(supportvec[i-2] == 1)[0]
        indices = indices.reshape(-1,1)
        zeros = np.zeros(indices.shape)
        event_type = np.full((indices.shape), i)
        events_arr.append(np.concatenate([indices, zeros, event_type], axis=1))

    events_arr = np.concatenate(events_arr)
    events_arr = events_arr.astype(int)

    return events_arr

"""
Sort events in a 3-column event array so that they're ordered in time. Run this after concatenating
the lists of events from triggers 3, 4, and 5.

events_arr: usually t_modeswitch
returns the same array but sorted

Example: events_inorder(t_modeswitch)
"""
def events_inorder(events_arr):
    indices = np.argsort(events_arr[:, 0])
    sorted_arr = events_arr[indices]
    return sorted_arr


""" 
Sorts the events found by mne into events from different soundcard channels.

events: array of all events after cleaning by clean_triggers

Returns separate arrays with only one event type each. 
Also returns a list of start times for each type of trial (listening, motor, error)

Example:

events_2, events_3, events_4, events_5, trial_starts = sort_events(events)
"""
def sort_events(events, clean = True):
    #assert 65282 and 65284 and 65288 and 65296 in events[:,2], "Not all trig categories are present"

    if 65282 and 65284 and 65288 and 65296 in events[:,2]:
        print("All event types present")    
    else:
        print("Some event types missing. Check data.")

    if clean == True:
        events_2 = clean_triggers(events[events[:,2] == 65282]) #t2 - keystrokes
        events_3 = clean_triggers(events[events[:,2] == 65284]) #t3
        events_4 = clean_triggers(events[events[:,2] == 65288]) #t4
        events_5 = clean_triggers(events[events[:,2] == 65296]) #t5 (it's also used for trial starts in subject 1, so there should be two far apart at the beginning)
        events_6 = clean_triggers(events[events[:,2] > 65296]) #t6

        #get only the start triggers that are at least 11min 10 secs (670 s) mins apart
        #motor and error trials are exactly 10 mins long. Passive listening is 11:05 mins.
        trial_starts = clean_triggers(events[events[:,2] == 65296], threshold = 1372160) 
    
    else:
        events_2 = events[events[:,2] == 65282]
        events_3 = events[events[:,2] == 65284]
        events_4 = events[events[:,2] == 65288]
        events_5 = events[events[:,2] == 65296]
        events_6 = events[events[:,2] > 65296]
        trial_starts = events[events[:,2] == 65298]

    return events_2, events_3, events_4, events_5, events_6, trial_starts


"""
New version of sort_events that takes into account the new MIDI trigger box
"""

#for the config where the MIDI trigger is in all channels
def sort_events_MIDI(events, clean = True):
    #assert 65282 and 65284 and 65288 and 65296 in events[:,2], "Not all trig categories are present"

    if 2 and 4 and 8 and 16 and 65280 in events[:,2]:
        print("All event types present")    
    else:
        print("Some event types missing. Check data.")

    if clean == True:
        events_2 = clean_triggers(events[events[:,2] == 2]) #t2 - keystrokes
        events_3 = clean_triggers(events[events[:,2] == 4]) #t3
        events_4 = clean_triggers(events[events[:,2] == 8]) #t4
        events_5 = clean_triggers(events[events[:,2] == 16]) #t5 (it's also used for trial starts in subject 1, so there should be two far apart at the beginning)
        events_6 = clean_triggers(events[events[:,2] == 4096]) #t6 (not used for anything in most subjects)
        events_MIDI = clean_triggers(events[events[:,2] ==65280])

        #get only the start triggers that are at least 11min 10 secs (670 s) mins apart
        #motor and error trials are exactly 10 mins long. Passive listening is 11:05 mins.
        
        
    #FIX THIS
    else:
        events_2 = events[events[:,2] == 2]
        events_3 = events[events[:,2] == 4]
        events_4 = events[events[:,2] == 8]
        events_5 = events[events[:,2] == 16]
        events_6 = events[events[:,2] == 4096]
        events_MIDI = events[events[:,2] == 65280]

    trial_starts = clean_triggers(events[events[:,2] == 16], threshold = 1372160) 


    return events_2, events_3, events_4, events_5, events_6, trial_starts, events_MIDI

#======================================================================================
#                       FINDING/CLASSIFYING KEYSTROKES
#======================================================================================

def find_sections(raw, section_trigggers, mode_triggers, downfreq = 128):
    """ 
    Finds timeframes of each playing mode in the error trial (inv, shinv, or norm)
    raw: eeg data
    section_triggers: the mode of interest
    t_modeswitch: events, defined beforehand. All the mode-related triggers

    Example: 
    find_sections(raw, t_shinv, t_modeswitch)
    """
    section_times = []
    for segment_start in section_trigggers[:,0]:

        #segment_start = time #because crop() uses seconds and not samples
        remaining_trigs =  [x for x in mode_triggers[:,0] if x > segment_start]
        if len(remaining_trigs) > 0: 
            segment_end = remaining_trigs[0]

        #make sure the max length of eeg is not exceeded
        else:
            segment_end = raw.times.max()*downfreq

        section_times.append([segment_start, segment_end])

    return np.array(section_times)



def find_keystrokes(raw, t_keystrokes, timeframes):
    """ 
    Finds the keystrokes that fall into a certain condition.
    raw: EEG data
    t_keystrokes: all keystroke events
    timeframes: the times of the condition you want, found with find_sections

    Example: find_keystrokes(raw, t_keystrokes, norm_times)
    """
    keystrokes = t_keystrokes[:,0]
    filt_keystrokes = []
    for segment in timeframes:
        start = segment[0]
        end = segment[1]

        condition = (keystrokes >= start) & (keystrokes <= end)
        filt_keystrokes.extend(keystrokes[condition])

    indices = np.where(np.isin(t_keystrokes[:, 0], filt_keystrokes))[0]
    filt_keystrokes_events = t_keystrokes[indices]
    return filt_keystrokes_events




# def mapchange_keystrokes(t_modeswitch, t_keystroke): #old version
#     """ 
#     Finds all the keystroke triggers that are the first keystrokes after a map change.

#     t_modeswitch: subset of events_array with all mode switch triggers
#     t_keystroke: subset of events_array with all keystrokes
#     ---
#     Returns: first keystrokes, a np array in the same format as events_array (3 columns, first column is time)
#     """
    
#     first_keystrokes = []
#     mode_times = t_modeswitch[:, 0]  # Extract mode switch times
#     mode_idx = 0
#     num_modes = len(mode_times)

#     for keystroke in t_keystroke:
#         k_time = keystroke[0]

#         # Advance mode_idx until the keystroke is after the current mode switch
#         while mode_idx < num_modes and k_time > mode_times[mode_idx]:
#             first_keystrokes.append(keystroke)  # Add the keystroke
#             mode_idx += 1  # Move to the next mode switch

#         # If the keystroke is earlier than the next mode switch, ignore it
#         if mode_idx < num_modes and k_time <= mode_times[mode_idx]:
#             continue


#     return np.array(first_keystrokes)

# def mapchange_keystrokes_2(t_modeswitch, t_keystroke): #other old version
#     """ 
#     Finds all the keystroke triggers that are the first keystrokes after a map change.

#     t_modeswitch: subset of events_array with all mode switch triggers
#     t_keystroke: subset of events_array with all keystrokes
#     ---
#     Returns: first keystrokes, a np array in the same format as events_array (3 columns, first column is time)
#     """
    
#     first_keystrokes = []
#     switch_times = t_modeswitch[:, 0]  # Extract mode switch times
#     switch_idx = 0
#     n_switches = len(switch_times)

    

#     keystroke_times = t_keystroke[:,0]

#     for keystroke in t_keystroke:
#         if switch_idx >= n_switches - 2:
#             break

#         ktime = keystroke[0]

#         #make sure the keystroke is between two mode switches
#         if ktime> switch_times[switch_idx] and ktime < switch_times[switch_idx+1]:
#             first_keystrokes.append(keystroke)
#             switch_idx+=1
        
#         #if there are no keystrokes between two mode switches, this forces it to jump to the next mode
#         elif ktime > switch_times[switch_idx+1] and ktime < switch_times[switch_idx+2]:
#             first_keystrokes.append(keystroke)
#             switch_idx+=2

#     return np.array(first_keystrokes)


def mapchange_keystrokes_4(t_modeswitch, t_keystroke):
    """ 
    Finds all the keystroke triggers that are the first keystrokes after a map change.

    t_modeswitch: subset of events_array with all mode switch triggers 
    t_keystroke: subset of events_array with all keystrokes
    ---
    Returns: first keystrokes, a np array in the same format as events_array (3 columns, first column is time)
    """
    
    first_keystrokes = []
    switch_times = t_modeswitch[:, 0]  # Extract mode switch times
    switch_idx = 0
    n_switches = len(switch_times)

    
    keystroke_times = t_keystroke[:,0]

    for keystroke in t_keystroke:
        if switch_idx >= n_switches - 2:  # Adjusted condition to avoid out-of-bounds
            break

        ktime = keystroke[0]

        # Make sure the keystroke is between two mode switches
        if ktime > switch_times[switch_idx] and ktime < switch_times[switch_idx + 1]:
            first_keystrokes.append(keystroke)
            switch_idx += 1

        # Skip consecutive mode switches until we find a keystroke in between
        else:
            try:
                while ktime > switch_times[switch_idx + 1]:
                    switch_idx += 1
            except IndexError:
                continue

            # If the keystroke is still valid after skipping switches, add it
            if ktime > switch_times[switch_idx] and ktime < switch_times[switch_idx + 1]:
                first_keystrokes.append(keystroke)
                switch_idx += 1

    return np.array(first_keystrokes)


def withinmap_keystrokes(t_keystrokes, first_keystrokes):
    """ 
    Finds the keystrokes EXCEPT the ones immediately after a map change 
        (by removing the keystrokes that appear in first_keystrokes from the array of all keystrokes, t_keystrokes)

    t_keystroke: subset of events_array with all keystrokes
    first_keystrokes: subset of t_keystroke with the keystrokes immediately after map change 
    ---
    Returns: other_keystrokes, a np array in the same format as events_array(3 columns, first column is time)
    
    """
    t_keystrokes_set = set(map(tuple, t_keystrokes))
    first_keystrokes_set = set(map(tuple, first_keystrokes))

    # Find the difference between the sets
    others_set = t_keystrokes_set - first_keystrokes_set

    # Convert back to a NumPy array
    other_keystrokes = np.array(list(others_set))
    other_keystrokes = other_keystrokes[other_keystrokes[:,0].argsort()]

    return other_keystrokes

def construct_ep_ev(reconst_raw, mode_keystrokes, erp_begin, erp_end):
    """ 
    Constructs epochs and evokeds given the types of keystrokes you want to use
    Interpolates bad channels
    ---
    returns: epochs, evokeds 
    """
    epochs_mode = mne.Epochs(reconst_raw, mode_keystrokes, tmin=erp_begin, tmax=erp_end, preload=True)
    epochs_mode = epochs_mode.copy().interpolate_bads(reset_bads = True)
    evoked_mode = epochs_mode.average()

    return epochs_mode, evoked_mode

def epochs_bymode(reconst_raw, t_keystrokes, t_mode, t_modeswitch, erp_begin, erp_end):
    """ 
    Use for finding keystrokes that belong to a particular mapping
    combines the steps of finding relevant sections, keystrokes, and creating epochs and evokeds
    
    ---
    returns: epochs, evokeds
    """
    mode_sections = find_sections(reconst_raw, t_mode, t_modeswitch)
    mode_keystrokes = find_keystrokes(reconst_raw, t_keystrokes, mode_sections)
    epochs_mode, evoked_mode = construct_ep_ev(reconst_raw, mode_keystrokes, erp_begin, erp_end)

    return epochs_mode, evoked_mode

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# STUFF BELOW NOT IN USE
#-------------------------------------------------------------------

"""
Concatenates different sections of the EEG experiment that are OF THE SAME LENGTH
(It does the same thing as Epochs sort of, but you end up with a raw object instead of an epoch object so you can work with it further...)
Useful for gathering all the recordings of muted and unmuted sections of the motor experiment. 

raw: the original raw file straight out of mne.io.read_raw_bdf
events: the events array of interest. Take only the first column, like tc_mute[:,0]
segment_dur: duration of the segment. For muted segments it's 30 seconds, for unmuted it's 10


Note: the time axis will still be continuous even if the data has been chopped up
"""
def concat_uniform(raw, events, segment_dur, fs):
    segments = []
    for time in events[:,0]:

        segment_start = time/fs #because crop() uses seconds and not samples
        segment_end = segment_start+segment_dur

        #make sure the max length of eeg is not exceeded
        if segment_end > raw.times.max():
            segment_end = raw.times.max()

        segment = raw.copy().crop(tmin = segment_start, tmax = segment_end)
        segments.append(segment)

    return mne.io.concatenate_raws(segments)


""" 
Concatenates different sections of the EEG experiment that are OF DIFFERENT LENGTHS
Useful for gathering all the recordings of the inv and norm mapping segments of error trials

raw: the original raw file straight out of mne.io.read_raw_bdf
events1: trigger array marking beginning of the segment of interest. Take only first column. Ex. trig_inv[:,0]
events2: trigger array marking beginning of other segments (and therefore the end of the segment of interest)
"""
def concat_nonuniform(raw, events1, events2, fs):
    segments = []
    for time in events1[:,0]:

        segment_start = time/fs #because crop() uses seconds and not samples
        remaining_trigs =  [x for x in events2[:,0] if x > time]
        if len(remaining_trigs) > 0: 
            segment_end = remaining_trigs[0]/fs

        #make sure the max length of eeg is not exceeded
        else:
            segment_end = raw.times.max()

        segment = raw.copy().crop(tmin = segment_start, tmax = segment_end)
        segments.append(segment)
        
    return mne.io.concatenate_raws(segments)


