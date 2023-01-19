import h5py
import numpy as np
import matplotlib.pyplot as plt
import neo
from quantities import ms, Hz, uV, s
from ceed.analysis import CeedDataReader
import quantities as pq
import pandas as pd
import os
import math
import random
from scipy.signal import hilbert
import itertools
from tqdm import tqdm
import sys

"""Quantify + visualize how well each neuron aligns to a rhythmic stimulus"""


def PPC(SP):

    """Calculates Pairwise Phase Coherence"""
   # a_deg = map(lambda x: np.ndarray.item(x), SP)
   # a_rad = map(lambda x: math.radians(x), a_deg)

   # a_rad = np.fromiter(a_rad, dtype=np.float)
    a_complex = map(lambda x: [math.cos(x), math.sin(x)], SP)

    all_com = list(itertools.combinations(a_complex, 2))
    dp_array = np.empty(int(len(SP) * (len(SP) - 1) / 2))

    pbar = tqdm(all_com, total=len(all_com), desc="Processing pairwise phase dot products...", file=sys.stdout)
    d = 0
    for combination in pbar:
        dp = np.dot(combination[0], combination[1])
        dp_array[d] = dp
        d += 1
    dp_sum = np.sum(dp_array)
    return dp_sum / len(dp_array)

def MI(spike_phase_hist, nbins):
    """Calculates Modulation index (can be biased by number of spikes)"""
    sph_norm = spike_phase_hist / np.sum(spike_phase_hist)
    divergence_kl = np.sum(
        sph_norm * np.log(sph_norm * nbins))
    return divergence_kl / np.log(nbins)

def custom_round(x, base=5, return_int=True):
    if return_int:
        return int(float(base) * round(float(x)/float(base)))
    else:
        return float(base) * round(float(x)/float(base))

def get_spike_phase_hist_onhilb(analytic_sig, spikes, sample_times, start, stop, nbins=18, plot=False, fs=1000):

    """
    Parameters
    ----------
    analytic_sig: hilbert transformed lfp
    spikes: list of Neo SpikeTrain objects
        A list of one or more SpikeTrain objects which we will calculate and plot time-resolved PLV measures for.
    start, stop: Quantity scalar with dimension time
        Start and stop times designating the time period for PLV calculation
        (Set empty to use entire signal)
    filter_range: iterable of two integer values
        Designates the low and high bounds for the bandpass filter

    Returns
    ----------
    ppc: float
        Pairwise phase consistency
    """
    phase_bins = np.linspace(-180, 180, nbins+1)
    start = round(start,3)
    if stop:
        stop = round(stop,3)

    # analytic_sig = analytic_sig[int(round(start*fs)):int(round(stop*fs))]
    instantaneous_phase = np.unwrap(np.angle(analytic_sig, deg=True))

    # samples_cycle = int(np.round((1/frequency) / (1/Fs)))
    # if samples_cycle < nbins:
    #     nbins = samples_cycle
    for spike_train in spikes:

        if type(spike_train) is not neo.core.spiketrain.SpikeTrain:
            raise TypeError('spikes is not a Neo.SpikeTrain object!')
        if stop:
            spike_train = spike_train.time_slice(start, stop - .001*s) #end time is inclusive

        all_phases = [None] * len(spike_train)
        all_phases_rand = [None] * len(spike_train)

        t = 0

        for n in spike_train:
            #find nearest sample
            idx = (np.abs(sample_times - n/s)).argmin()
            ip_of_spike = instantaneous_phase[idx]
            all_phases[t] = ip_of_spike[0]
            n = random.uniform(0, np.max(spike_train)/s)
            idx = (np.abs(sample_times - n)).argmin()
            ip_of_spike = instantaneous_phase[idx]
            all_phases_rand[t] = ip_of_spike
            t += 1

        a_deg = all_phases
        a_rad = [math.radians(x) for x in all_phases]

        a_cos = [math.cos(x) for x in a_rad]
        a_sin = [math.sin(x) for x in a_rad]
        a_cos, a_sin = np.fromiter(a_cos, dtype=np.float64), np.fromiter(a_sin, dtype=np.float64)
        try:
            uv_x = sum(a_cos) / len(a_cos)
            uv_y = sum(a_sin) / len(a_sin)
            uv_radius = np.sqrt((uv_x * uv_x) + (uv_y * uv_y))
            uv_phase = np.angle(complex(uv_x, uv_y))
            # sig = 100 * (1. - np.exp(-1 * len(all_phases) * (uv_radius ** 2)))
            pval = np.exp(-1 * len(all_phases) * (uv_radius ** 2))
        except:
            pval = np.nan
            uv_radius = np.nan

        dig_phases = np.digitize(all_phases, phase_bins, nbins + 1)
        dig_phases_rand = np.digitize(all_phases_rand, phase_bins, nbins + 1)
        spike_phase_hist = np.zeros(nbins+1)
        spike_phase_hist_rand = np.zeros(nbins+1)
        for bin in range(0, nbins + 1):
            spike_phase_hist[bin] = np.sum(dig_phases == bin)
            spike_phase_hist_rand[bin] = np.sum(dig_phases_rand == bin)

        if plot:
            bins = np.linspace(-np.pi, np.pi, nbins + 1)

            plt.bar(bins, spike_phase_hist,
                    width=bins[1] - bins[0],
                    bottom=0.0)
            """calc MI"""
            norm_phase_hist = spike_phase_hist[1:] - spike_phase_hist_rand[1:]
            norm_phase_hist += np.abs(np.min(norm_phase_hist)) + 1
            MI_norm = MI(spike_phase_hist[1:], nbins)
            ppc = PPC(all_phases)
            plt.subplot(3,1,1)
            plt.title(pval)

        return spike_phase_hist, uv_radius, pval

def plot_cosine(exp_sr_df, unit, segment=[0,1], frequency=20):

    plt.figure(figsize=(18, 10))
    plt.subplot(3,1,1)
    curr_df = exp_sr_df[exp_sr_df['Unit #']==unit]
    unit_i = 0
    for i, row in curr_df.iterrows():
        plt.plot(row['event spike rate'] - row['t_start'] * s,
                 unit_i * np.ones_like(row['event spike rate']), '|', markersize=4)
        unit_i += 1
        plt.xlim(segment[0], segment[1])
        # plt.ylim(-1, len(intensities)*2+1)
    plt.axvline(0)
    plt.subplot(3,1,2)
    Fs = 119.96*12
    times = np.arange(Fs*10)
    amp = np.cos(((2 * np.pi * frequency * times+np.pi*Fs)) / Fs)*.5+.5
    plt.plot(np.linspace(0,10,len(amp)), amp)
    plt.xlim(segment)
    plt.subplot(3,1,3, polar=True)


def run():
    segment = [0, 5]
    ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-08-05\\'
    fname = 'slice6_merged.h5'
    try:
        os.mkdir(ffolder+'Figures\\stim_spike_locking\\'+fname)
    except:
        print('dir already made')
    units2plt = []
    rec_fname = fname.replace('.h5','')
    spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
    sk_df = pd.read_pickle(ffolder + 'Analysis\\spyking-circus\\' + rec_fname + '.pkl')

    exp_df = pd.read_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')

    ceed_data = ffolder+fname
    Fs=20000
    reader = CeedDataReader(ceed_data)
    # open the data file
    reader.open_h5()
    all_d = []
    for i, row in sk_df.iterrows():
        # if cluster_info[unit + 1][5] == 'good':
        unitST = row['Data']  # / Fs
        tstop = exp_df['t_start'].iloc[-1]*s+10*s
        if unitST[-1] > tstop:
            tstop = unitST[-1] + 2
        neo_st = neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                          t_stop=tstop)
        if not row['Group'] == 'noise':
            if len(units2plt) == 0 or row['ID'] in units2plt:
                for row_i in range(0,exp_df.shape[0]):
                    start = exp_df.iloc[row_i]['t_start']*s+segment[0]*s
                    end = exp_df.iloc[row_i]['t_start']*s+segment[1]*s
                    try:
                        seg_st = neo_st.time_slice(start, (end - .001*s) + .001 * pq.s)
                        d = {
                            'event spike rate': np.squeeze(seg_st),
                            'evoked n_spikes': len(seg_st),
                            'Unit #': row['ID'],
                            'Group': row['Group'],
                            't_start': exp_df.iloc[row_i]['t_start'],
                            'substage': exp_df.iloc[row_i]['substage'],
                            'loop': exp_df.iloc[row_i]['loop'],
                            # 'Intensity': exp_df.iloc[row_i]['Intensity'],
                            'Experiment': exp_df.iloc[row_i]['Experiment'],
                            # 'Frequency': exp_df.iloc[row_i]['Noisy param'],
                            # 'duration': exp_df.iloc[row_i]['duration']
                        }
                        all_d.append(d)
                    except ValueError:
                        print('Stimuli after recording stopped')

    ev_sr_df = pd.DataFrame(all_d)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    frequencies = [2, 10, 20, 35, 55, 90, 140, 200]
    # frequencies = ev_sr_df['Frequency'].unique()
    # frequencies = np.sort(frequencies)
    # frequencies = frequencies[np.where(~np.isnan(frequencies))]

    dicts = []
    # exp_sr_df = ev_sr_df[ev_sr_df['substage'] == 'Vary_frequency stage']
    exp_sr_df = ev_sr_df
    nbins = 18
    sph_dict = {}
    for frequency in frequencies:
        sph_dict['SPH_'+str(frequency)] = np.zeros(nbins+1)
    for unit in exp_sr_df['Unit #'].unique():
        unit_df = exp_sr_df[exp_sr_df['Unit #'] == unit]
        try:
            os.mkdir(ffolder+'Figures\\stim_spike_locking\\'+fname+'\\'+str(unit)+'\\')
        except:
            x=1 #just a line of code for the sake of it
        for frequency in frequencies:
            # curr_df = unit_df[unit_df['Frequency'] == frequency]
            curr_df = unit_df[unit_df['substage'] == 'Cos-'+str(frequency)+'Hz stage']
            unit_i = 0
            spike_times = []
            for i, row in curr_df.iterrows():
                #append spike times together
                try:
                    for spike in row['event spike rate']:
                        spike_times.append(spike - row['t_start']*s)
                except:
                    spike_times.append(row['event spike rate'] - row['t_start'] * s)
            if len(spike_times) > 0:
                st = neo.core.SpikeTrain(spike_times * s, t_stop=np.max(spike_times * s))
                Fs = 119.96 * 12
                samples = np.arange(Fs * 10)
                amp = np.cos(((2 * np.pi * frequency * samples + np.pi * Fs)) / Fs)
                neo_signal = neo.core.AnalogSignal(amp, units=uV,
                                                   sampling_rate=Fs * Hz, t_start=0 * s)
                analytic_signal = hilbert(neo_signal, None, 0)
                sample_times = np.arange(0, 10, 1/Fs)
                plot_cosine(curr_df, unit, segment, frequency)
                sph, uv_radius, pval = get_spike_phase_hist_onhilb(analytic_signal, [st], sample_times, start=0 * s, stop=np.max(st), nbins=18, plot=True, fs=Fs)
                sph_dict['SPH_'+str(frequency)] += sph
                plt.savefig(ffolder+'Figures\\stim_spike_locking\\'+fname+'\\'+str(unit)+'\\'+str(unit)+'_'+str(frequency)+'Hz.png')
                dicts.append({'Slice':fname, 'Unit':unit, 'Frequency':frequency, 'uv_radius':uv_radius, 'pval':pval, 'suprise':-np.log(pval), 'group':curr_df.iloc[0]['Group']})
                plt.close()
    df = pd.DataFrame(dicts)
    df.to_excel(ffolder+'Figures\\stim_spike_locking\\'+fname+'\\df.xlsx')

    plt.figure(figsize=(18.7,6))
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    for i, frequency in enumerate(frequencies):
        plt.subplot(1,len(frequencies), i+1, polar=True)
        plt.bar(bins, sph_dict['SPH_'+str(frequency)],
                width=bins[1] - bins[0],
                bottom=0.0)
        plt.title(frequency)
    plt.savefig(ffolder + 'Figures\\stim_spike_locking\\' + fname + '\\' +'overall_sphs.png')


def plot_dataframe():
    import seaborn as sns
    ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-08-05\\'
    fname = 'slice6_merged.h5'
    df = pd.read_excel(ffolder+'Figures\\stim_spike_locking\\'+fname+'\\df.xlsx')
    plt.figure()
    sns.barplot(data=df, x='Frequency', y='uv_radius')
    plt.show()
    plt.figure()
    for i, row in df.iterrows():
        df.at[i, 'suprise'] = -np.log(row['pval'])
    sns.pointplot(data=df, x='Frequency', y='suprise')
    plt.show()


run()


