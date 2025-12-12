"""
Functions for visualizing analyses
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from stats_utils import * 

"""
Plots the sections in time corresponding to each playing mode: inv, shinv and norm. Each is shaded a different colour.
section_list: the list of start and end times of a certain type of section. Found using the function find_sections in pp_utils

example: plot_sections(inv_sections)
"""
def plot_sections(section_list, downfreq = 128):
    plt.figure(figsize=(20,10))

    y1 = 0
    y2 = -1
    colors = ['green', 'orange', 'red']
    colour_idx = 0

    for sections in section_list:
        y = [y1, y1, y2, y2]
        sections = sections/downfreq 
        for i in range(sections.shape[0]):
            x = [sections[i][0], sections[i][1], sections[i][1], sections[i][0]]
            plt.fill(x, y, color=colors[colour_idx], alpha=0.2)
        colour_idx +=1
        y1 -=1
        y2-=1

    
    plt.xlabel('Time (s)')
    plt.ylabel('Playing mode')
    #plt.show()


def find_mean_sem_evs(epochs_df, ch_idx, col_to_ave):

    """ 
    Finds the mean and sem of evoked so that they can be plotted
    epochs_df: dataframe with epochs, subject, etc info
    ch_idx: a SINGLE channel index
    col_to_ave: name of column in dataframe where the evokeds are stored
    """
    mean_ev = epochs_df[col_to_ave].mean(axis=0)[ch_idx].flatten()
    sem_ev = sem(np.stack(epochs_df[col_to_ave].values), axis=0)[ch_idx].flatten()
    
    return mean_ev, sem_ev

def find_mean_sem_eps(epochs_df, ch_idx, col_to_ave='epochs', data = 'epochs'):
    """ 
    Takes a dataframe with columns 'subject', 'period', 'musician', 'epochtype', 'epochs'
    Returns mean and sem of epochs for each subject
    """
    # evoked_col = f'{col_to_ave}_evokeds'
    
    if 'evokeds' not in epochs_df.columns:

        epochs_df['evokeds'] = epochs_df[col_to_ave].apply(lambda x: np.mean(x, axis=0))
    
    mean_ev = epochs_df['evokeds'].mean(axis=0)[ch_idx].flatten()
    sem_ev = sem(np.stack(epochs_df['evokeds'].values), axis=0)[ch_idx].flatten()
    
    return mean_ev, sem_ev

def plot_mean_sem(times, mean, sem, color = 'grey', label = None, linewidth = 1):
    """ 
    Plots ERP with sem bars
    """
    plt.plot(times, mean, color=color, label=label, linewidth=linewidth)
    plt.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.15)

def config_erp_plot(axis_fontsize=20, tick_fontsize=15, linewidth = 1):
    """ 
    Basic config of axes and ticks for ERP plots
    """
    plt.xlabel('Time (s)', fontsize=axis_fontsize)
    plt.ylabel('Amplitude (µV)', fontsize=axis_fontsize)
   
    
    # Draw reference lines
    plt.hlines(0, -0.5, 0.5, color='black', linewidth=linewidth)
    plt.vlines(0, -6.5e-6, 5e-6, color='black', linewidth=linewidth)
    
    # Format y-axis values in µV instead of scientific notation
    plt.ticklabel_format(style='plain', axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1e6:.1f}'))
    
    # Set axis limits
    plt.xlim(-0.1, 0.45)
    plt.ylim(-6.5e-6, 5e-6)
    
    # Set tick font sizes
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    
    # Set legend font size
    plt.legend(fontsize=tick_fontsize)
    
def diff_score_df(data_df, scores_df, sub_col_name, data_columns): 

    """ 
    combines the df with training scores and the df with some participant-specific data to allow corr analyses

    Note: subject is the unformatted column. sub is formatted as strings with leading 0s, e.g. '02'
    power_diff_df: expected columns: sub, diff
    scores_df: exported from training repo with scores from all subjects. Expected columns:  subject, score, musician, rank
    sub_col_name: name of subject column in data df (usually sub or subject)
    data_columns: columns from the data
    ---
    returns diff_df_ranked: df with sub, training score, data of interest, and relative ranking to other subjects
    """

    #configure sub column of scores_df to make it match the column in the diff df
    scores_df['sub'] = scores_df['subject'].astype(int).apply(lambda x: f'{x:02d}')

    #configure data df
    data_df['sub'] = data_df[sub_col_name]
    assert data_df['sub'].apply(lambda x: isinstance(x, str) and len(x) == 2).all(), "Subjects should be expressed as a string with 2 characters"

    data_filt_df = data_df[['sub'] + data_columns]
    data_filt_df.reset_index(drop=True)

    diff_df_ranked = data_filt_df.merge(scores_df, on =['sub'])
    #power_diff_df_ranked = power_diff_df_ranked[['sub', 'diff', 'musician', 'score', 'rank']]

    return diff_df_ranked


def plot_ci95(distr, bar_ypos, bracketsize = 3, **kwargs):

    ci_l, ci_u = ci95_bs(distr, axis = 0)
    plt.hlines(bar_ypos, ci_l, ci_u, **kwargs)
    plt.vlines(ci_l, bar_ypos - bracketsize, bar_ypos + bracketsize, **kwargs)
    plt.vlines(ci_u, bar_ypos - bracketsize, bar_ypos + bracketsize, **kwargs)
    

def config_bs_plot(bar_ypos, **kwargs):
    #plot vertical line at 0
    plt.vlines(0, 0, bar_ypos, **kwargs)

    #plt.legend(fontsize = xtick_fontsize)
    plt.ylim(0, bar_ypos+10)
    plt.xlabel(r'Mean difference ($\mu$V)')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1e6:.1f}'))  # Convert to µV
    plt.yticks([0, 120])
    # plt.xticks(fontsize = fontsize*0.7)
