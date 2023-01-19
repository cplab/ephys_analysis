import os
import pandas as pd
import neo
import elephant
import numpy as np
import quantities as pq
from spectrogram import plot_spectrogram
import mne
import matplotlib.pyplot as plt
import scipy

"""Plot evoked spectrograms for spike trains"""

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-08-05\\'
rec_fname = 'slice1_merged'

sampling_period = 1 * pq.ms #in ms
sk_df = pd.read_pickle(ffolder + 'Analysis\\spyking-circus\\' + rec_fname + '.pkl')
Fs = int(1000/float(sampling_period))
info = mne.create_info(ch_names=['1'], sfreq=Fs, ch_types=['eeg'])
segment = (-.1, .1)
baseline = (-0, 0)
freqs = np.arange(1, 150, 2)  # define frequencies of interest
n_cycles = freqs / 2
average = True

exp_df = pd.read_pickle(ffolder + 'Analysis\\' + rec_fname + '.h5_exp_df.pkl')
init = True
for evt in range(0, exp_df.shape[0]):
    if init:
        my_annot = mne.Annotations(onset=exp_df.iloc[evt]['t_start'],
                                   duration=.5,
                                   description=exp_df.iloc[evt]['substage'])
        init = False
    else:
        my_annot += mne.Annotations(onset=exp_df.iloc[evt]['t_start'],
                                   duration=.5,
                                   description=exp_df.iloc[evt]['substage'])


# drug_log = pd.read_excel(ffolder+'Analysis\\drug log.xlsx', sheet_name=rec_fname.replace('_merged',''))

for i, row in sk_df.iterrows():
    if row['Group'] == 'good':

        unit1 = row['Data']
        unit = row['ID']
        ST_overall = neo.SpikeTrain(unit1* pq.s, t_stop=(np.max(unit1) + 1) * pq.s)
        neo_st1 = elephant.statistics.instantaneous_rate(ST_overall, sampling_period=sampling_period,
                                                         kernel=elephant.kernels.GaussianKernel(1 * pq.ms))
        ST_bin = np.squeeze(np.asarray(neo_st1))

        raw = mne.io.RawArray(np.expand_dims(ST_bin,0), info)
        raw.set_annotations(my_annot)
        events, event_ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_ids['Whole experiment'], tmin=segment[0], tmax=segment[1],
                            event_repeated='drop',
                            baseline=None)
        tmp_pow = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                average=average, picks=epochs.picks)
        # tmp_pow.apply_baseline(mode='zscore', baseline=(baseline[0] - segment[0], baseline[1] - segment[0]))

        file_pow = tmp_pow.data
        plt.close()
        if not average:
            # try:
            #     os.mkdir(savedir)
            # except:
            #     print('directory already exists')
            # for trial in range(0, tmp_pow.data.shape[0]):
            #     plt.figure(figsize=(18, 12))
            #     plt.imshow(np.squeeze(tmp_pow.data[trial, 0, :, :]),
            #                extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
            #                aspect='auto', origin='lower', interpolation='none')
            #     plt.close()
                # plt.savefig(savedir + 'trial' + str(trial))
            plt.figure(figsize=(18, 12))
            plt.imshow(np.squeeze(np.mean(tmp_pow.data[:, 0, :, :],0)),
                       extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                       aspect='auto', origin='lower', interpolation='none', cmap='jet')
        else:
            plt.figure(figsize=(18, 12))
            fig = plt.imshow(np.mean(tmp_pow.data, (0)),
                             extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
                             aspect='auto', origin='lower', interpolation='bicubic', cmap='jet', vmin=0, vmax=200000)
            plt.xlim(-.5, 1)
            if row['Group'] == 'good':
                plt.savefig(ffolder + 'Figures\\Spike spectrograms\\' + rec_fname + '\\event\\unit_' + str(
                    unit) + '_morlet_good')
            else:
                plt.savefig(
                    ffolder + 'Figures\\Spike spectrograms\\' + rec_fname + '\\event\\unit_' + str(unit) + '_morlet')

            # for i in range(tmp_pow.data.shape[0]):
            #     plt.close()
            #     fig = plt.imshow(np.squeeze(tmp_pow.data[i, :, :]),
            #                      extent=[tmp_pow.times[0], tmp_pow.times[-1], tmp_pow.freqs[0], tmp_pow.freqs[-1]],
            #                      aspect='auto', origin='lower', interpolation='none', vmin=0, vmax=5)
            #     # plt.savefig(slicedir + '\\pow' + experiment + '_perelec\\' + elecs2plot[i] + '.png')