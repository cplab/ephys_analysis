import numpy as np
import neo
from quantities import uV, Hz, ms, s
import matplotlib.pyplot as plt
from slice_analysis.lfp_analyses.filters import butter_lowpass_filter, iir_notch, butter_highpass_filter
import scipy.signal as signal
from ceed.analysis import CeedDataReader


all_electrodes = [
    'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
    'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
    'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
    'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
    'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
    'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
    'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12',
    'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11',
    'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10',
    'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]
column_letters = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "J": 9, "K": 10, "L": 11, "M": 12}


def mea_plot_traces(h5_file, times=None, notch=True, resample=False, hp=True, skip_electrodes=[], ylim=None):
    """
    Function that plots voltage traces based on their position on the multielectrode array, similar to how they are
    observed during recordings with the MCS software.


    Parameters
    ------------
    h5_file: h5 file
        An h5 file that has been merged from the MCS and Ceed h5 files, and contains the electrode data
    times: list or tuple, default=None
        A list containing two time quantities, indicating the start and stop times which will be used to slice the
        voltage data prior to plotting. If None, then the full time series will be plotted.
    notch: bool, default=True
        If True, the data will be notched at 60*Hz to remove line noise.
    resample: bool, default=True
        If True, the data will be resampled at 512*Hz, default=True
    hp: bool, default=True
        If True, the data will be highpass filtered
        `matplotlib.specgram()` for options.
    skip_electrodes: list
        List containing the IDs of the electrode which should be skipped (i.e., not plotted)
    ylim: list or tuple, default=None
        A list or tuple containing the lower and upper bounds of the y-axis to plot. This is fed directly to plt.ylim()

    """
    reader = CeedDataReader(h5_file)
    reader.open_h5()
    reader.load_application_data()
    reader.load_mcs_data()

    if skip_electrodes:
        electrodes = list(set(all_electrodes) - set(skip_electrodes))
        electrodes.sort()
    else:
        electrodes = all_electrodes

    _, axes = plt.subplots(
        nrows=12, ncols=12, sharex=True, sharey=True)

    for electrode in electrodes:
        print("Processing electrode " + electrode + "...")
        offset, scale = reader.get_electrode_offset_scale(electrode)
        fs = reader.electrodes_metadata[electrode]['sampling_frequency'] * Hz
        raw_data = (np.array(reader.electrodes_data[electrode]) - offset) * scale
        raw_data = raw_data * 2000000
        electrode_signal = neo.core.AnalogSignal(raw_data, sampling_rate=fs, units='uV')

        if times is not None:
            start, stop = times[0], times[1]
            electrode_signal = electrode_signal.time_slice(start, stop)

        if resample:
            rfs = 512*Hz
            num_samples = int(electrode_signal.duration.rescale(s) * rfs / Hz)
            electrode_signal = butter_lowpass_filter(electrode_signal, rfs/2, fs)
            electrode_signal = signal.resample(electrode_signal, num_samples)
            electrode_signal = neo.core.AnalogSignal(electrode_signal, units=uV, sampling_rate=rfs, t_start=0*s)
            print("Resampled signal")

        if notch:
            current_fs = electrode_signal.sampling_rate
            electrode_signal = iir_notch(electrode_signal, current_fs, 60, axis=0, quality=60)
            electrode_signal = neo.core.AnalogSignal(electrode_signal, units=uV, sampling_rate=current_fs, t_start=0*s)
            print("Notched signal")

        if hp:
            current_fs = electrode_signal.sampling_rate
            electrode_signal = butter_highpass_filter(electrode_signal, 5, current_fs, axis=0)
            electrode_signal = neo.core.AnalogSignal(electrode_signal, units=uV, sampling_rate=current_fs, t_start=0*s)
            print("Highpass filtered signal")

        coordinates = [x for x in electrode]
        x = column_letters[coordinates[0]] - 1
        if len(coordinates) == 2:
            y = int(coordinates[1]) - 1
        else:
            y = int(coordinates[1] + coordinates[2]) - 1

        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(hspace=.001, wspace=.001)

        axes[y][x].plot(electrode_signal.times.rescale(ms), electrode_signal, 'k', linewidth=0.5)

    blackout = [(1, 1), (1, 2), (1, 3), (1, 10), (1, 11), (1, 12), (2, 1), (2, 2), (2, 11), (2, 12), (3, 1), (3, 12),
                (10, 1), (10, 12), (11, 1), (11, 2), (11, 11), (11, 12), (12, 1), (12, 2), (12, 3), (12, 10), (12, 11),
                (12, 12)]
    for coord in blackout:
        x, y = coord[0]-1, coord[1]-1
        rect = plt.Rectangle((0, -1000), (stop-start).rescale(ms).item(), 2000, facecolor='k')
        axes[y][x].add_patch(rect)
    if times is not None:
        plt.xlim([0, (stop-start).rescale(ms).item()])
    if ylim is not None:
        plt.ylim(ylim)
    plt.sca(axes[1, 1])
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=6)
    plt.show()
