#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def generate_noise_template(
    num_samples,
    sample_rate,
    noise_type = "gaussian",
    rms = 0.1,
    bandwidth = None,
    seed = None
    ):
    ''' Utiity for generating noise templates.
    '''
    
    # Seed the randomness?
    if seed is not None:
        np.random.seed(seed)
    
    ## Time-domain noise generation ##
    # What type of noise is requested?
    if noise_type == "gaussian":
        noise = np.random.normal(0, 1, num_samples)
    elif noise_type == "white":
        noise = np.random.uniform(-1, 1, num_samples)
    elif noise_type == "pink":
        # Simple 1/f approach using FFT shaping.
        freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
        spectrum = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
        spectrum /= np.sqrt(freqs + 1e-6)
        noise = np.fft.irfft(spectrum, num_samples)
    else:
        raise ValueError("Invalid noise type. Legal arguments are 'gaussian', 'white', or 'pink'.")
    
    # Shape bandwidth?
    if bandwidth is not None:
        freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
        spectrum = np.fft.rfft(noise)
        
        # Create frequency mask, and filter out anything broader
        # in the spectrum than the user needs.
        half_bw = bandwidth / 2
        mask = np.abs(freqs) <= half_bw  # Note: centered around baseband!
        spectrum *= mask
        noise = np.fft.irfft(spectrum, num_samples)
    
    # Normalise with RMS?
    noise *= rms / np.sqrt(np.mean(noise**2))
    
    # Return!
    return noise
