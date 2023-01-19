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

"""Plot spectrograms from spike trains"""

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-08-05\\'
rec_fname = 'slice1_merged'

# binsize = 1 #in ms
sk_df = pd.read_pickle(ffolder + 'Analysis\\spyking-circus\\' + rec_fname + '.pkl')

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

try:
    os.mkdir(ffolder+'Figures\\Spike spectrograms\\')
except:
    print('Spike spectrograms folder already made')

try:
    os.mkdir(ffolder+'Figures\\Spike spectrograms\\'+rec_fname)
except:
    print('recording folder already made')

# units = [1, 6, 7, 13, 16, 18, 41, 46, 41, 65, 79, 86, 90, 81, 124, 130, 135, 137, 145, 146, 149, 150, 155]
# for unit in units:
#     curr_df = sk_df[sk_df['ID']==unit]
# drug_log = pd.read_excel(ffolder+'Analysis\\drug log.xlsx', sheet_name=rec_fname.replace('_merged',''))
exp_df = pd.read_pickle(ffolder + 'Analysis\\' + rec_fname+'.h5' + '_exp_df.pkl')

for i, row in sk_df.iterrows():
    # if not row['Group'] == 'noise':
    if row['Group'] == 'good':
        unit1 = row['Data']
        unit = row['ID']
        # unit1 = sk_df[sk_df['ID']==unit]['Data'].iloc[0]
        ST_overall = neo.SpikeTrain(unit1* pq.s, t_stop=(np.max(unit1) + 1) * pq.s)
        # ST_bin = elephant.conversion.BinnedSpikeTrain(ST_overall, bin_size=binsize * pq.ms, t_start=0 * pq.s)
        neo_st1 = elephant.statistics.instantaneous_rate(ST_overall, sampling_period=.1 * pq.ms,
                                                         kernel=elephant.kernels.GaussianKernel(1 * pq.ms))
        ST_bin = np.squeeze(np.asarray(neo_st1))

        freqs, times, Sxx = scipy.signal.spectrogram(ST_bin - np.mean(ST_bin), 10000, window='hann', nperseg=3000,
                                                     noverlap=2500)

        plt.figure(figsize=(18, 10))
        plt.imshow(Sxx[:60,:],
                   extent=[0, ST_bin.shape[0] / 10000, freqs[0], freqs[60]],
                   aspect='auto', origin='lower', interpolation='bicubic', cmap='jet', vmin=0, vmax=100)
        ax = plt.gca()

        # for i, log_row in drug_log.iterrows():
        #     time = log_row['Time - minutes']*60 + log_row['Time - seconds']
        #     plt.axvline(time)
        #     ax.text(time, 125, log_row['Event'], ha="left", va="top", rotation=0, size=8)
        add_stim_to_spectrogram_df(exp_df)
        if row['Group'] == 'good':
            plt.savefig(ffolder+'Figures\\Spike spectrograms\\'+rec_fname+'\\unit_'+str(unit)+'_good')
        else:
            plt.savefig(ffolder+'Figures\\Spike spectrograms\\'+rec_fname+'\\unit_'+str(unit))

        plt.close()