import os
import numpy as np
import pickle

from scipy.signal import convolve
from commpy.filters import rrcosfilter, rcosfilter


# Parameters for Root Raise Cosine Signal
rolloff = 0.5
Fs = 25e6
oversample_factor = 16
Ts = oversample_factor/Fs
tVec, sPSF = rrcosfilter(oversample_factor*8, rolloff, Ts, Fs)
tVec, sPSF = tVec[1:], sPSF[1:]
sPSF = sPSF.astype(np.complex64)

seg_len = int(2**15 + 2**13)
n_sym = seg_len//oversample_factor

def generate_srrc_signal(n_sym=n_sym, oversample_factor=oversample_factor, sPSF=sPSF, Fc=0, Fs=Fs):
    sQ = 1./np.sqrt(2)*(np.random.randn(n_sym) + 1j*np.random.randn(n_sym))
    sQ_padded = np.zeros(len(sQ)*oversample_factor, dtype=np.complex64)
    start_idx = oversample_factor//2
    sQ_padded[start_idx::oversample_factor] = sQ

    sig = convolve(sQ_padded, sPSF, 'same') # Waveform with PSF
    sig *= np.exp(2*np.pi*1j*np.arange(len(sig))*Fc/Fs, dtype=np.complex64)
    return sig, sQ_padded, sQ, None

def get_psf():
    return sPSF

def matched_filter(sig, sPSF=sPSF, Fc=0, Fs=Fs):
    sig_filt = sig*np.exp(-2*np.pi*1j*np.arange(len(sig))*Fc/Fs, dtype=np.complex64)
    sig_filt = convolve(sig_filt, sPSF/np.sum(sPSF), 'same')
    return sig_filt
