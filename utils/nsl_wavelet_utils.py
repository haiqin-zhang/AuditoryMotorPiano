import numpy as np
def wavelet_freqs(fs, n_freqs):
    
    nyquist_freq = fs / 2
    freqs = np.arange(1, n_freqs + 1)

    # Calculate the frequencies corresponding to each index
    frequencies = nyquist_freq * (2 ** (-(n_freqs - freqs) / 24)) #maybe change the 24 for future applications
    return frequencies



def find_closest_indices(array, values):
    """
    Find indices of the closest elements in a 1D array to given values.
    
    Parameters:
    - array (np.ndarray): A 1D numpy array to search.
    - values (list): A list of values to find closest elements for.
    
    Returns:
    - list: A list of indices in the array corresponding to the closest elements.
    """
    indices = []
    for value in values:
        # Get the index of the closest value
        closest_index = (np.abs(array - value)).argmin()
        indices.append(closest_index)
    return indices
