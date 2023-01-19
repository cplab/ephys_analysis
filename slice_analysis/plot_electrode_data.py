from ceed.analysis import CeedDataReader
import read_experiments
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import mne
import matplotlib.pyplot as plt
import pandas as pd

"""Use MNE's plotting functions to plot electrode data, and scroll"""

fname = 'slice3_merged.h5'
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-06-02\\'
segment = (-4,8)
baseline=(-3, -1)
Fs = 20000
rfs = 1000
filtered = False
ceed_data = ffolder+fname

# create instance that can load the data
reader = CeedDataReader(ceed_data)
print('Created reader for file {}'.format(reader.filename))
# open the data file
reader.open_h5()
if filtered:
    lfp_data = np.load(ffolder+'filt_and_rsamp\\' + fname +'.npy',allow_pickle=True)
else:
    from ceed.analysis import CeedDataReader
    reader = CeedDataReader(ffolder+fname)
    reader.open_h5()
    reader.load_mcs_data()

# Init data for mne
# Read in relevant experiment times
exp_df = pd.read_pickle(ffolder + 'Analysis\\' + fname + '_exp_df.pkl')
my_annot = read_experiments.convert_expdf_toMNE(exp_df, Fs)

"""Plot all electrodes"""
# elecs2plot = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
# #elecCombo = ['F2','F6']
# info = mne.create_info(elecs2plot, sfreq=rfs)
#
# mne_data = np.empty((len(elecs2plot), reader.electrodes_data.item()['B5'].shape[0]))
# i = 0
# for elec in elecs2plot:
#     mne_data[i, :] = reader.electrodes_data.item()[elec]
#     i += 1
#
# raw = mne.io.RawArray(mne_data, info)
# raw.set_annotations(my_annot)
# raw.plot(duration=10, n_channels=120, scalings='auto', title='Data')
# print('plot')

"""Plot half at a time"""
# elecs2plot = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
elecs2plot = ['B6', 'D4', 'D7', 'E6', 'H6', 'J8', 'K9']
# elecCombo = ['F2','F6']
if filtered:
    info = mne.create_info(elecs2plot, sfreq=rfs)
else:
    info = mne.create_info(elecs2plot, sfreq=Fs)
if filtered:
    mne_data = np.empty((len(elecs2plot), lfp_data.item()['B5'].shape[0]))
    i = 0
    for elec in elecs2plot:
        mne_data[i, :] = lfp_data.item()[elec]
        i += 1
else:
    mne_data = np.empty((len(elecs2plot), reader.electrodes_data['B5'].shape[0]))
    i = 0
    for elec in elecs2plot:
        mne_data[i, :] = np.asarray(reader.electrodes_data[elec])
        i += 1

raw = mne.io.RawArray(mne_data, info)
raw.set_annotations(my_annot)
raw.plot(duration=.1, n_channels=120, scalings='auto', title='Data')
print('plot')

elecs2plot = ['G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
#elecCombo = ['F2','F6']
info = mne.create_info(elecs2plot, sfreq=rfs)

mne_data = np.empty((len(elecs2plot), reader.electrodes_data.item()['B5'].shape[0]))
i = 0
for elec in elecs2plot:
    mne_data[i, :] = reader.electrodes_data.item()[elec]
    i += 1

raw = mne.io.RawArray(mne_data, info)
raw.set_annotations(my_annot)
raw.plot(duration=10, n_channels=120, scalings='auto', title='Data')
print('plot')


"""Plot average of all electrodes"""
info = mne.create_info(['Average'], sfreq=rfs)

mne_data = np.empty((1, reader.electrodes_data.item()['B5'].shape[0]))
for elec in elecs2plot:
    mne_data[0,:] += reader.electrodes_data.item()[elec]

mne_data[0,:] = mne_data[0,:] / len(elecs2plot)

raw = mne.io.RawArray(mne_data, info)
raw.set_annotations(my_annot)
raw.plot(duration=10, n_channels=33, scalings='auto', title='Data')


plt.figure()
plt.plot(mne_data[0,0:10000])