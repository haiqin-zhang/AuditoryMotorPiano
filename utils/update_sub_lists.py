import os
from scipy.io import loadmat, savemat

"""
Usage: good_listen_subjects, good_motor_subjects,good_error_subjects,musicians,nonmusicians = load_subject_lists_LME()
"""

all_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

good_listen_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
good_motor_subjects = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '18', '19', '20'] 
    #02 doesn't have enough data due to crashes
    #17 post looks like a listening ERP
    #21 has strange artifacts

good_error_subjects = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    #02 doesn't have enough data due to crashes 
    #03 data doesn't look like an ERP
    #21 has strange artifacts

nonmusicians = ['01', '03', '04', '05', '08', '09', '10', '11', '16', '19', '20']
musicians = ['02', '06', '07', '12', '13', '14', '15', '17', '18', '21']

savemat('subject_lists.mat', 
        {
            'good_listen_subjects':good_listen_subjects,
            'good_motor_subjects': good_motor_subjects,
            'good_error_subjects':good_error_subjects,
            'nonmusicians': nonmusicians,
            'musicians':musicians

        })

def load_subject_lists(): #this one doesn't have good error subjects list
    """ 
    Use load_subject_lists_LME instead but this function still present in many files
    Returns lists of subjects sorted by task and musicianship
    """
    subject_lists = loadmat('subject_lists.mat')
    good_listen_subjects = subject_lists['good_listen_subjects'].tolist()
    good_motor_subjects = subject_lists['good_motor_subjects'].tolist()
    musicians = subject_lists['musicians'].tolist()
    nonmusicians = subject_lists['nonmusicians'].tolist()

    return good_listen_subjects, good_motor_subjects, musicians, nonmusicians

def load_subject_lists_LME():
    """ 
    Same as above but includes a list for error subjects
    
    """
    subject_lists = loadmat('subject_lists.mat')
    good_listen_subjects = subject_lists['good_listen_subjects'].tolist()
    good_motor_subjects = subject_lists['good_motor_subjects'].tolist()
    good_error_subjects = subject_lists['good_error_subjects'].tolist()
    musicians = subject_lists['musicians'].tolist()
    nonmusicians = subject_lists['nonmusicians'].tolist()

    return good_listen_subjects, good_motor_subjects, good_error_subjects, musicians, nonmusicians


