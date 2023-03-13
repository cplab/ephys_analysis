import numpy as np
import neo
from quantities import uV, Hz, ms, s
import h5py
from filters import butter_lowpass_filter
import scipy
from ceed.analysis import CeedDataReader
import sklearn


all_electrodes = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9',
            'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
            'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
            'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
            'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
            'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
            'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12',
            'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11',
            'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9',
            'M4', 'M5', 'M6', 'M7', 'M8', 'M9']


def pca_denoise(h5_file, denoised_h5_file, channels=all_electrodes, rfs=512*Hz):
    """
    Uses Principal component analysis (PCA) to denoise the resampled datasets to remove 60 Hz interference and
    other highly correlated noise sources.

    Parameters
    ------------
    h5_file: h5 file
        An h5 file that has been merged from the MCS and Ceed h5 files, and contains the electrode data
    denoised_h5_file: str
        String indicated where to save the denoised H5 file
    channels: list
        The channels which to perform the denoising on
    rfs: quantity
        Resampling rate
    """
    reader = CeedDataReader(h5_file)
    reader.open_h5()
    reader.load_application_data()
    reader.load_mcs_data()

    offset, scale = reader.get_electrode_offset_scale(channels[0])
    fs = reader.electrodes_metadata[channels[0]]['sampling_frequency'] * Hz
    first = True

    for i, channel in enumerate(channels):
        print("Loaded electrode " + channel + "...")
        raw_data = (np.array(reader.electrodes_data[channel]) - offset) * scale
        raw_data = raw_data * 2000000
        raw_signal = neo.core.AnalogSignal(raw_data, sampling_rate=fs, units='uV')

        num_samples = int(raw_signal.duration.rescale(s) * rfs / Hz)
        resampled_signal = butter_lowpass_filter(raw_signal, rfs/2, fs)
        resampled_signal = scipy.signal.resample(resampled_signal, num_samples)
        resampled_signal = neo.core.AnalogSignal(resampled_signal, units=uV, sampling_rate=rfs)

        if first:
            original_data = np.zeros([num_samples, len(channels)])
            first = False

        original_data[:, i] = resampled_signal.reshape(resampled_signal.shape[0])

    mu = np.mean(original_data, axis=-1)
    pca = sklearn.decomposition.PCA()
    pca.fit(original_data)

    print("Performing PCA...")
    nComp = 1
    Xhat = np.dot(pca.transform(original_data)[:, :nComp], pca.components_[:nComp, :])
    mu_reshaped = [mu] * len(channels)
    mu_reshaped = np.vstack(tuple(mu_reshaped)).T
    Xhat += mu_reshaped

    transformed_data = original_data - Xhat
    hf = h5py.File(denoised_h5_file, 'w')

    for i, channel in enumerate(channels):
        hf.create_dataset(channel, data=transformed_data[:, i])
    hf.close()
