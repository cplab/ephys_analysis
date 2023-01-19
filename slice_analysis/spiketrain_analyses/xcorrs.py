import h5py
import numpy as np
import matplotlib.pyplot as plt
import more_spike_lfp_scripts
import neo
import math
from scipy.stats import circmean
import scipy.stats as stats
from scipy.signal import hilbert, resample
from quantities import ms, Hz, uV, s
from ceed.analysis import CeedDataReader
import csv
import read_experiments
import elephant
import quantities as pq
import pandas as pd
import pickle
from itertools import combinations

"""Autocorrs, Xcorrs, over recording period"""

def xcorr(exp_df, st1, st2, offset, duration, window, bin, evoked=False, remove_artifact=True, general_range=[]):
    """
    Compute xcorr

    Evoked: segment by time around stimuli; false to just do overall xcorr
    """

    all_odor_st1, all_odor_st2 = [], []
    if len(general_range)==2:
        st1 = st1[np.where(np.logical_and(st1 > general_range[0], st1 < general_range[1]))[0]]
        st2 = st2[np.where(np.logical_and(st2 > general_range[0], st2 < general_range[1]))[0]]
    if evoked:
        for _, experiment in exp_df.iterrows():
            t_start = experiment['t_start']
            t_start = float(t_start) * s
            odor_st1 = st1[np.where(np.logical_and(st1>t_start/pq.s + offset, st1<t_start/pq.s + offset + duration))[0]]
            odor_st2 = st2
            odor_st1 = [t.item() for t in odor_st1]
            odor_st2 = [t.item() for t in odor_st2]
            all_odor_st1 += odor_st1
            all_odor_st2 += odor_st2
    else:
        odor_st1 = [t.item() for t in st1]
        odor_st2 = [t.item() for t in st2]
        all_odor_st1 += odor_st1
        all_odor_st2 += odor_st2

    low, high = -window, window
    histo_window = (low, high)
    histo_bins = int((high - low) / bin)
    all_odor_st2_arr = np.asarray(all_odor_st2)
    all_odor_diffs = []
    for st1_st in all_odor_st1:
        window_start, window_stop = st1_st - window, st1_st + window
        in_window_idxs = np.where(np.logical_and(all_odor_st2_arr>=window_start, all_odor_st2_arr<=window_stop))[0]
        if len(in_window_idxs) > 0:
            in_window = all_odor_st2_arr[in_window_idxs]
            # in_window = all_odor_st2[in_window_idxs[0]:in_window_idxs[-1]+1]
            for st2_st in in_window:
                spike_diff = (st2_st - st1_st)
                if not remove_artifact or not np.abs(spike_diff) < .00005:
                    all_odor_diffs.append(spike_diff)
    counts, bins = np.histogram(all_odor_diffs, bins=histo_bins, range=histo_window)
    return counts, bins

def ISIs_per_spike(ev_sr_df, sts, units, duration=1*s, offset=0*s, event='A strong sq', window=.1, bin=.001):

    """Compute Inter-Spike-Intervals (ISIs)"""
    exp_df = ev_sr_df[ev_sr_df['description']==event]
    for i, unit1 in enumerate(units):

            st1 = sts[unit1]
            ISIs_overall = []
            for _, experiment in exp_df.iterrows():
                t_start = experiment['Event time']
                t_start = float(t_start) * s
                odor_st1 = st1.time_slice(t_start+offset, t_start+offset+duration)
                ISIs = []
                for spi in range(0,len(odor_st1)):
                    if spi > 0:
                        ISIs.append(odor_st1[spi] - odor_st1[spi-1])
                ISIs_overall.append(ISIs)
    return ISIs_overall


import os
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-07-18\\'
Fs = 20000
for file in os.listdir(ffolder):
    if 'slice4_merged' in file:
        try:
            os.mkdir(ffolder+r'\Figures\xcorrs\\'+file)
        except:
            print('dir already made')
        try:
            os.mkdir(ffolder + r'\Figures\autocorrs\\' + file)
        except:
            print('dir already made')

        sk_df = pd.read_pickle(ffolder + 'Analysis\\spyking-circus\\' + file.replace('.h5', '') + '.pkl')
        exp_df = pd.read_pickle(ffolder + 'Analysis\\' + file+'_exp_df.pkl')


        binsize = .002
        window = .05
        good_units = []
        for i, row in sk_df.iterrows():
            if row['Group']=='good':
                good_units.append(row['ID'])

        """xcorr for all combos of good neurons"""
        savedir = ffolder + r'\Figures\xcorrs\\' + file
        for combo in combinations(good_units, 2):
            unit1 = combo[0]
            unit2 = combo[1]
            if not unit1 == unit2:
                plt.figure(figsize=(9, 5))
                st1 = sk_df[sk_df['ID'] == unit1]['Data'].iloc[0]
                # st1 = st1[np.where(np.logical_and(st1>segment[0], st1<segment[1]))]
                st2 = sk_df[sk_df['ID'] == unit2]['Data'].iloc[0]
                # st2 = st2[np.where(np.logical_and(st2 >segment[0], st2 < segment[1]))]
                # counts, bins = xcorr([], st1=sk_df[sk_df['ID']==unit1]['Data'].iloc[0], st2=sk_df[sk_df['ID']==unit2]['Data'].iloc[0], offset=[], duration=[], window=.1, bin=binsize, evoked=False)
                counts, bins = xcorr(exp_df=exp_df, st1=st1, st2=st2, offset=[], duration=[], window=window, bin=binsize, evoked=False, remove_artifact=True)
                plt.bar(bins[:-1] * 1000, counts, width=binsize * 1000, linewidth=1.2, edgecolor='k', align='edge')
                plt.xlabel('Time (ms)')
                plt.savefig(savedir + '\\unit' + str(unit1) + '_unit' + str(unit2) + '_binsize' + str(binsize) + '_window' + str(window) + '.png')
                plt.close()

        """xcorr for one neuron in comparison to all other good neurons"""
        sel_units = [129]
        savedir = ffolder + r'\Figures\xcorrs\\' + file
        for unit1 in sel_units:
            for unit2 in good_units:
                if not unit1==unit2:
                    plt.figure(figsize=(9, 5))
                    st1 = sk_df[sk_df['ID'] == unit1]['Data'].iloc[0]
                    # st1 = st1[np.where(np.logical_and(st1>segment[0], st1<segment[1]))]
                    st2 = sk_df[sk_df['ID'] == unit2]['Data'].iloc[0]
                    # st2 = st2[np.where(np.logical_and(st2 >segment[0], st2 < segment[1]))]
                    # counts, bins = xcorr([], st1=sk_df[sk_df['ID']==unit1]['Data'].iloc[0], st2=sk_df[sk_df['ID']==unit2]['Data'].iloc[0], offset=[], duration=[], window=.1, bin=binsize, evoked=False)
                    counts, bins = xcorr(exp_df=exp_df, st1=st1, st2=st2, offset=[], duration=[], window=window, bin=binsize, evoked=False, remove_artifact=True)
                    plt.bar(bins[:-1] * 1000, counts, width=binsize * 1000, linewidth=1.2, edgecolor='k', align='edge')
                    plt.xlabel('Time (ms)')
                    plt.savefig(savedir + '\\unit' + str(unit1) + '_unit' + str(unit2) + '_binsize' + str(binsize) + '_window' + str(window) + '.png')
                    plt.close()

        "Autocorrs"
        savedir = ffolder + r'\Figures\autocorrs\\' + file
        for unit in good_units:
            # for di, drug in drug_log_df.iterrows():
            st1 = sk_df[sk_df['ID'] == unit]['Data'].iloc[0]
            st2 = sk_df[sk_df['ID'] == unit]['Data'].iloc[0]
            counts, bins = xcorr(exp_df=exp_df, st1=st1, st2=st2, offset=0, duration=1, window=window, bin=binsize, evoked=False, remove_artifact=True,
                                 general_range=[])
            plt.bar(bins[:-1] * 1000, counts, width=binsize * 1000, linewidth=1.2, edgecolor='k', align='edge')
            plt.xlabel('Time (ms)')
            # plt.savefig(savedir + '\\unit' + str(unit) + '_binsize' + str(binsize) + '_window' + str(window) + '_'
            #             + drug['Event'] + str(di) + '.png')
            plt.savefig(savedir + '\\unit' + str(unit) + '_binsize' + str(binsize) + '_window' + str(window) + '_' + '.png')
            plt.close()


#
"""Do xcorrs of all neurons in comparison to 1"""
# unit1 = 7
# for i, unit in enumerate(sts):
#     counts, bins = xcorr([], st1=sts[unit1], st2=unit, offset=[], duration=[], window=.1, bin=binsize, evoked=False)
#     plt.figure()
#     plt.bar(x=bins[0:-1],
#             height=counts,
#             width=binsize,
#             align='edge')
#     plt.savefig(r'C:\Users\Michael\Analysis\myRecordings_extra\21-07-21\Figures\spikes\xcorrs\\slice5\\compw7\\'+str(i))
#     plt.close()


