from scipy.signal import butter, filtfilt, firls, iirnotch, cheby2
from scipy import signal
import numpy as np


def iir_notch(data, fs, frequency, quality=15., axis=-1):

    norm_freq = frequency/(fs/2)
    b, a = iirnotch(norm_freq, quality)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y


def custom_round(x, base=5, return_int=True):
    if return_int:
        return int(np.float(base) * round(np.float(x)/np.float(base)))
    else:
        return np.float(base) * round(np.float(x)/np.float(base))


def custom_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5, axis=-1):
    nyq = fs * 0.5
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y


#this is mimicking the filter used in Ray et al., 2011
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bp_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis, padtype=None)
    return y


def butter_bandstop(lowstop, highstop, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowstop/nyq, highstop/nyq
    b, a = butter(order, [low,high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = butter_bandstop(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=axis, padtype=None)
    return y


def firls_bp_filter(data, lowcut, highcut, fs, axis=-1, trans_width = 0.15):
    nyquist = fs/2.
    filt_order = float(round(3*(fs/float(lowcut))))
    if not (filt_order % 2):
        filt_order +=1
    ffreqs  = [0, (1-trans_width)*lowcut, lowcut, highcut, (1+trans_width)*highcut, nyquist]
    ideal = [0, 0, 1, 1, 0, 0]
    b = firls(filt_order, ffreqs, ideal, nyq=nyquist)
    w, h = signal.freqz(b, [1])
    return filtfilt(b, [1], data, axis=axis, padtype=None)


def cheby2_bp_filter(data, low, high, fs, order=5, rs=40, rs_pad=5, axis=-1):
    nyquist = 0.5 * fs
    rsf_low, rsf_high = (low-rs_pad)/nyquist, (high+rs_pad)/nyquist
    rsf = [np.float(rsf_low), np.float(rsf_high)]  # rsf contains the frequencies at which signal is attenuated to rs
    b, a = cheby2(order, rs, rsf, 'bandpass')
    y = filtfilt(b, a, data, axis=axis, padtype=None)
    return y


def fir_filter(data, order, cutoff, fs):
    firwin = signal.firwin(order, cutoff, width=None, window='hamming', pass_zero=True, scale=True)
    return filtfilt(firwin, [1], data, axis=0)
