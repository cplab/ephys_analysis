import h5py
import numpy as np
import matplotlib.pyplot as plt
import neo
from quantities import ms, Hz, uV, s
from ceed.analysis import CeedDataReader
import quantities as pq
import pandas as pd
import os
import seaborn as sns

"""
Plot rasters, often to compare responses under different conditions

Lots of different examples of comparisons I've plotted. Hopefully this is a good starting point for whatever comparisons you will plot
"""

def plot_square_cosine():
    """Plot responses to each odor, presented individually"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_cosine\\')
    except:
        print('dir already made')

    individ_shapes = ['Square stage', 'Cos stage']
    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='Square cosine test']

    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #'] == unit]
        for si, shape in enumerate(individ_shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.subplot(len(individ_shapes), 1, si + 1)
            unit_i = 0
            trialcount = 0
            for oi, row in curro_df.iterrows():
                plt.plot(row['event spike rate'] - row['t_start'] * s,
                         unit_i * np.ones_like(row['event spike rate']), '|', markersize=4, color=colors[si])
                unit_i += 1
                trialcount += 1
                plt.xlim(segment[0], segment[1])
            plt.ylim(-.5, trialcount+.5)
            if si < len(individ_shapes) - 1:
                plt.xticks([])
            plt.axvline(0)
            plt.title(shape)
        # plt.savefig(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_cosine\\' + str(unit)+'_'+str(row['Group'])+'.png')
        plt.savefig(ffolder + r'\Figures\rasters\\' + rec_fname + '\\time_to_response\\' + str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_square_test():
    """Plot responses square stim"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_test\\')
    except:
        print('dir already made')

    individ_shapes = ['Square stage']
    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='Square test']

    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #'] == unit]
        for si, shape in enumerate(individ_shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.subplot(len(individ_shapes), 1, si + 1)
            unit_i = 0
            trialcount = 0
            for oi, row in curro_df.iterrows():
                plt.plot(row['event spike rate'] - row['t_start'] * s,
                         unit_i * np.ones_like(row['event spike rate']), '|', markersize=4, color=colors[si])
                unit_i += 1
                trialcount += 1
                plt.xlim(segment[0], segment[1])
            plt.ylim(-.5, trialcount+.5)
            if si < len(individ_shapes) - 1:
                plt.xticks([])
            plt.axvline(0)
            plt.title(shape)
        plt.savefig(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_test\\' + str(unit)+'_'+str(row['Group'])+'.png')
        # plt.savefig(ffolder + r'\Figures\rasters\\' + rec_fname + '\\time_to_response\\' + str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_square_pulse():
    """Plot responses square pulse (5ms) stim"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_pulse\\')
    except:
        print('dir already made')

    individ_shapes = ['Square pulse stage']
    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='Square test']

    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #'] == unit]
        for si, shape in enumerate(individ_shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.subplot(len(individ_shapes), 1, si + 1)
            unit_i = 0
            trialcount = 0
            for oi, row in curro_df.iterrows():
                plt.plot(row['event spike rate'] - row['t_start'] * s,
                         unit_i * np.ones_like(row['event spike rate']), '|', markersize=4, color=colors[si])
                unit_i += 1
                trialcount += 1
                plt.xlim(segment[0], segment[1])
            plt.ylim(-.5, trialcount+.5)
            if si < len(individ_shapes) - 1:
                plt.xticks([])
            plt.axvline(0)
            plt.title(shape)
        plt.savefig(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_pulse\\' + str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_square_pulse_habituation():
    """Plot responses to repeated presentation of a square pulse"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_pulse_habituation\\')
    except:
        print('dir already made')

    individ_shape = 'Square pulse stage'
    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='Square test']

    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #'] == unit]
        shape = individ_shape
        curro_df = curr_df[curr_df['substage'] == shape]
        loops = exp_sr_df['loop'].unique()
        for li, loop in enumerate(loops):
            plt.subplot(len(loops), 1, li + 1)
            unit_i = 0
            trialcount = 0
            currl_df = curro_df[curro_df['loop']==loop]
            for oi, row in currl_df.iterrows():
                plt.plot(row['event spike rate'] - row['t_start'] * s,
                         unit_i * np.ones_like(row['event spike rate']), '|', markersize=2, color='black')
                unit_i += 1
                trialcount += 1
                plt.xlim(segment[0], segment[1])
            plt.ylim(-.5, trialcount+.5)
            plt.ylabel(loop)
        if li < len(loops) - 1:
            plt.xticks([])
        plt.axvline(0)
        plt.savefig(ffolder + r'\Figures\rasters\\' + rec_fname + '\\square_pulse_habituation\\' + str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_by_intensity():
    """Compare responses to stimuli presented at different intensities"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\byintensity\\')
    except:
        print('dir already made')

    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='varyintensity_squarefunc']
    intensities = exp_sr_df['Intensity'].unique()
    intensities = np.sort(intensities)
    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #']==unit]
        unit_i = 0
        for int_i, intensity in enumerate(intensities):
            curri_df = curr_df[curr_df['Intensity']==intensity]
            for i, row in curri_df.iterrows():
                plt.plot(row['event spike rate'] - row['t_start'] * s,
                         unit_i * np.ones_like(row['event spike rate']), '|', markersize=4, color=colors[int_i])
                unit_i += 1
                plt.xlim(segment[0], segment[1])
                # plt.ylim(-1, len(intensities)*2+1)
        plt.axvline(0)
        # plt.savefig(ffolder+r'\\Figures\rasters\\'+rec_fname+'\\byintensity\\'+str(unit)+'_'+str(row['Group'])+'.png')
        plt.savefig(ffolder+r'\\Figures\\time2response\\'+str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_cosine(frequency=20):
    """Plot response to a cosine stimulation"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\'+str(frequency)+'Hzcosine\\')
    except:
        print('dir already made')

    exp_sr_df = ev_sr_df[ev_sr_df['substage']=='Vary frequency stage']
    exp_sr_df = exp_sr_df[exp_sr_df['Noisy param']==frequency]
    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        plt.subplot(2,1,1)
        curr_df = exp_sr_df[exp_sr_df['Unit #']==unit]
        unit_i = 0
        for i, row in curr_df.iterrows():
            plt.plot(row['event spike rate'] - row['t_start'] * s,
                     unit_i * np.ones_like(row['event spike rate']), '|', markersize=4)
            unit_i += 1
            plt.xlim(segment[0], segment[1])
            # plt.ylim(-1, len(intensities)*2+1)
        plt.axvline(0)
        plt.subplot(2,1,2)
        Fs = 120*4
        times = np.arange(Fs*10)
        amp = np.cos(((2 * np.pi * frequency * times+np.pi*Fs)) / Fs)*.5+.5
        plt.plot(np.linspace(0,10,len(amp)), amp)
        plt.xlim(segment)
        plt.savefig(ffolder+r'\\Figures\rasters\\'+rec_fname+'\\'+str(frequency)+'Hzcosine\\'+str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_spot_location():
    """Plot responses to spots presented at different locations"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\spot_location\\')
    except:
        print('dir already made')

    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='Test_spotlocation']
    individ_shapes = ['Square stage', 'Square stage glomerular layer']

    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #'] == unit]
        for si, shape in enumerate(individ_shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.subplot(len(individ_shapes), 1, si + 1)
            unit_i = 0
            for i, row in curro_df.iterrows():
                plt.plot(row['event spike rate'] - row['t_start'] * s,
                         unit_i * np.ones_like(row['event spike rate']), '|', markersize=4, color='blue')
                unit_i += 1
                plt.xlim(segment[0], segment[1])
                # plt.ylim(-1, len(intensities)*2+1)
            plt.title(shape)
            plt.axvline(0, color='black')
        plt.savefig(ffolder+r'\\Figures\rasters\\'+rec_fname+'\\spot_location\\'+str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')

def plot_vary_duration():
    """Plot responses to stimuli of different durations"""
    try:
        os.mkdir(ffolder + r'\Figures\rasters\\' + rec_fname + '\\varyduration\\')
    except:
        print('dir already made')

    exp_sr_df = ev_sr_df[ev_sr_df['Experiment']=='varyduration_squarefunc']
    durations = exp_sr_df['duration'].unique()
    durations = np.sort(durations)
    for unit in exp_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = exp_sr_df[exp_sr_df['Unit #']==unit]
        unit_i = 0
        for dur_i, duration in enumerate(durations):
            if not np.isnan(duration):
                plt.subplot(len(durations)-1,1,dur_i+1)
                curri_df = curr_df[curr_df['duration']==duration]
                for i, row in curri_df.iterrows():
                    plt.plot(row['event spike rate'] - row['t_start'] * s,
                             unit_i * np.ones_like(row['event spike rate']), '|', markersize=4, color=colors[dur_i])
                    unit_i += 1
                    plt.xlim(segment[0], segment[1])
                    # plt.ylim(-1, len(intensities)*2+1)
            plt.axvline(0)
            plt.axvline(duration)
        plt.axvline(0)
        plt.savefig(ffolder+r'\\Figures\rasters\\'+rec_fname+'\\varyduration\\'+str(unit)+'_'+str(row['Group'])+'.png')
        plt.close('all')


segment = [0, .05] #start time, end time around event to plot
ffolder = r''
fname = '_merged.h5'
units2plt = [] #if not filled, plot all; if filled, only plot units in list
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

    #Make a dataframe, containing each neurons response for each stimuli
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
                        'loop': exp_df.iloc[row_i]['sub_loop'],
                        # 'Noisy param': exp_df.iloc[row_i]['Noisy param'],
                        'Intensity': exp_df.iloc[row_i]['Intensity'],
                        'Experiment': exp_df.iloc[row_i]['Experiment'],
                        # 'duration': exp_df.iloc[row_i]['duration']
                    }
                    all_d.append(d)
                except ValueError:
                    print('Stimuli after recording stopped')

ev_sr_df = pd.DataFrame(all_d)

try:
    os.mkdir(ffolder+r'\Figures\rasters\\'+rec_fname)
except:
    print('dir already made')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plot_by_intensity()









