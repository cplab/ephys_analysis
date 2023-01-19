from spectrogram import plot_spectrogram
from ceed.analysis import CeedDataReader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import quantities as pq
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch

"""For each merged file, plot overall recording spectrogram for each electrode and save"""

def add_stim_to_spectrogram(reader):
    # get experiment data, add patches to plot
    for exp in range(0, len(reader.experiments_in_file)):
        reader.load_experiment(exp)
        if reader.experiment_stage_name != 'eyfp':
            shapes2plot = []
        else:
            shapes2plot = None
        if reader.electrode_intensity_alignment is not None:
            # find the peak of the stimulus
            for shape in reader.shapes_intensity.keys():
                if shape in shapes2plot or shapes2plot is None:
                    peak_idxs = np.where(reader.shapes_intensity[shape][:, 2] == 1)
                    # for i in range(0, peak_idxs[0].shape[0]):
                    #     idx = peak_idxs[0][i]
                    i = 0
                    while i < peak_idxs[0].shape[0]:
                        idx = peak_idxs[0][i]
                        if not idx >= reader.electrode_intensity_alignment.shape[0]:
                            ax = plt.gca()
                            t_start = reader.electrode_intensity_alignment[idx] / 20000
                            duration = .166
                            rect = matplotlib.patches.Rectangle((t_start, 0), duration, Fs / 2, linewidth=1,
                                                                edgecolor='k',
                                                                facecolor='none')
                            ax.add_patch(rect)
                            bbox_props = dict(boxstyle="round", fc='white', lw=0.5)
                            # ax.text(t_start, 225, shape, ha="left", va="top", rotation=0, size=8, bbox=bbox_props)
                        # i += 30  # skipping the number of other peaks in the same stimulus
                        i+=1

def add_stim_to_spectrogram_df(exp_df):
    # get experiment data, add patches to plot
    for evt in range(0,exp_df.shape[0]):
        ax = plt.gca()
        # rect = matplotlib.patches.Rectangle((exp_df.iloc[evt]['t_start'], 0), +.5, Fs / 2, linewidth=1,
        #                                     edgecolor='k',
        #                                     facecolor='none')
        # ax.add_patch(rect)
        plt.axvline(exp_df.iloc[evt]['t_start'], color='black', lw=.2)
        bbox_props = dict(boxstyle="round", fc='white', lw=0.5)
        ax.text(exp_df.iloc[evt]['t_start']/pq.s, 225, exp_df.iloc[evt]['substage'], ha="left", va="top", rotation=0, size=8, bbox=bbox_props)

def plot_spectrograms_batch(fdir, filtered):

    """
    For each merged file, plot overall recording spectrogram for each electrode and save.

    :param fdir: directory containing merged files
    :param filtered: If true, plot filtered data. Loads filtered .npy files from a subfolder of fdir 'filt_and_rsamp'
    """

    try:
        ffolder = fdir+'Figures\\'
        os.mkdir(ffolder)
    except:
        print('Figures folder already made')
    try:
        ffolder = ffolder+'Spectrograms\\'
        os.mkdir(ffolder)
    except:
        print('Spectrograms folder already made')


    filtered = True #If filtered is true,
    for fname in os.listdir(fdir):
        if '_merged.h5' in fname:

            ceed_data = fdir+fname
            Fs=1000
            Fs_raw = 20000
            reader = CeedDataReader(ceed_data)
            print('Created reader for file {}'.format(reader.filename))
            # open the data file
            reader.open_h5()
            saveloc = ffolder+fname+'\\'
            try:
                os.mkdir(saveloc)
            except:
                print('Files figure directory already exists')
            if filtered:
                reader.electrodes_data = np.load(fdir+'filt_and_rsamp\\' + fname +'.npy',allow_pickle=True)
            else:
                from ceed.analysis import CeedDataReader
                reader = CeedDataReader(fdir + fname)
                reader.open_h5()
                reader.load_mcs_data()
            exp_df = pd.read_pickle(fdir + 'Analysis\\' + fname + '_exp_df.pkl')

            elecs2plot = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
            for elec in elecs2plot:
                if filtered:
                    plot_spectrogram(reader.electrodes_data.item()[elec],Fs)

                    ax = plt.gca()
                else:
                    plot_spectrogram(reader.electrodes_data[elec], Fs_raw)
                add_stim_to_spectrogram_df(exp_df)
                plt.title(elec)
                # figManager = plt.get_current_fig_manager()
                # figManager.window.showMaximized()
                # plt.show()
                plt.savefig(saveloc + elec + '.png')
                plt.close()


