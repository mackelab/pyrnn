import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt

def PC_traj(state_traj, n_comp=3):
    """
    Projects (typical high dimensional) currents on (typical low dimensional) PC space

    args:
        state_traj, numpy array of size (batch_size, seq_len, n_rec)
    n_comp:
        number of PC components to project on

    returns:
        z: trajectories, numpy array of size (batch_size, seq_len, n_comp)
        varexp: variance explained by each component
    """

    [batch_size, seq_len, n_rec] = state_traj.shape

    # typically we simply 'append' all trials, treating all time steps as a sample
    # and units as features
    state_traj_btxd = np.reshape(state_traj, (batch_size * seq_len, n_rec))

    pca = PCA(n_components=n_comp)
    pca.fit(state_traj_btxd)
    varexp = pca.explained_variance_ratio_
    z = np.zeros((batch_size, seq_len, n_comp))

    # project on n_comp first PC components
    for batch_idx in range(batch_size):
        x_idx = state_traj[batch_idx]
        z[batch_idx] = pca.transform(x_idx)
    return z, varexp


def wrap(x):
    """returns angle with range [-pi, pi]"""
    return np.arctan2(np.sin(x), np.cos(x))


def complex_wavelet(timestep, freq, cycles, kernel_length=5):
    """
    Create wavelet of a certain frequency

    Args:
        timestep: simulation timestep in seconds
        freq: frequency of the wavelet
        cycles: number of oscillations of wavelet
        kernel_length: adapted per frequency
    Note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    gauss_sd = cycles / (2 * np.pi * freq)
    t = np.arange(0, kernel_length * gauss_sd, timestep)
    t = np.r_[-t[::-1], t[1:]]
    gauss = (1 / (np.sqrt(2 * np.pi) * gauss_sd)) * np.exp(
        -(t ** 2) / (2 * gauss_sd ** 2)
    )
    sine = np.exp(2j * np.pi * freq * t)
    wavelet = gauss * sine * timestep

    return wavelet


def inst_phase(sign, kernel, t, f, ref_phase=True, mode="same"):

    """
    Calculate instaneous angle and magnitude at certain frequency

    Args:
        sign: signal to analyse
        kernel: convolve with this, e.g. complex wavelet
        t: array of timesteps
        f: frequency to extract
        ref_phase: calculate phase with respect to sinusoid
        mode: convolution mode (see numpy.convolve)
    Returns:
        phase: phase at each timestep
        amp: amplitude at each timestep

    Note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    conv = np.convolve(sign, kernel, mode=mode)

    # cut off more in case kernel is too long
    if len(conv) > len(sign):
        st = (len(conv) - len(sign)) // 2
        conv = conv[st : st + len(sign)]
    amp = np.abs(conv)
    arg = np.angle(conv)
    if ref_phase:
        ref = wrap(2 * np.pi * f * t)
        phase = wrap(arg - ref)
    else:
        phase = arg
    return phase, amp


def scalogram(sign, cycles, t, timestep, freqs, kernel_length=5):

    """
    Create a scalogram of a signal, using complex wavelets

    Args:
        sign: signal to analyse
        t: array of timesteps
        timestep: timestep in seconds
        freqs: list of frequencies to extract
        timestep: simulation timestep in seconds
        cycles: number of oscillations of wavelet
        kernel_length: adapted per frequency

    Returns:
        phasegram: phase at each timestep, for each frequency
        ampgram: amplitude at each timestep, for each frequency

    Note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    phasegram = np.zeros((len(freqs), len(t)))
    ampgram = np.zeros((len(freqs), len(t)))
    for i, f in enumerate(freqs):
        kernel = complex_wavelet(timestep, f, cycles, kernel_length)
        phase, amp = inst_phase(sign, kernel, t, f)
        phasegram[i] = phase
        ampgram[i] = amp * np.sqrt(2)
    return phasegram, ampgram

def orth_proj(a,b, ret_comp = False):
    """
    Orthogonal projection of b on a
    Args:
        a in Nx1
        b in Nx1
    Returns:
        b_par in Nx1
        b_orth in Nx1
    """
    alpha = torch.matmul(a.t(), b)/torch.matmul(a.t(), a)
    b_par = alpha*a
    
    if ret_comp:
        b_orth = b - b_par
        return b_par, b_orth, alpha
    else:
        return b_par, alpha
    
    
def color_scheme():
    """
    Returns colorscheme for Liebe et al. (in preperation) 2022
    
    Returns:
        pltcolors: warm colorscheme
        pltcolors_alt: cold colorscheme
    """

    # red to yellow
    pltcolors = [
        [c / 255 for c in [255, 201, 70, 255]],
        [c / 255 for c in [253, 141, 33, 255]],
        [c / 255 for c in [227, 26, 28, 255]],
        [c / 255 for c in [142, 23, 15, 255]],
    ]
    hexcolors = str([mpl.colors.to_hex(pltcolors[i])[1:] for i in range(4)])

    # green blue
    pltcolors_alt = [
        [c / 255 for c in [161, 218, 180, 255]],
        [c / 255 for c in [65, 182, 196, 255]],
        [c / 255 for c in [34, 94, 168, 255]],
        [c / 255 for c in [10, 30, 69, 255]],
    ]

    a_file = open("matplotlibrc", "r")
    list_of_lines = a_file.readlines()
    list_of_lines[-1] = "axes.prop_cycle: cycler('color'," + hexcolors + ")"
    a_file = open("matplotlibrc", "w")
    a_file.writelines(list_of_lines)
    a_file.close()

    return pltcolors, pltcolors_alt
