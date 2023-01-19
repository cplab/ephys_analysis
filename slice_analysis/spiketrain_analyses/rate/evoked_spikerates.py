import numpy as np
import matplotlib.pyplot as plt
import neo
from quantities import ms, Hz, uV, s
from ceed.analysis import CeedDataReader
import elephant
import quantities as pq
import pandas as pd
import os
import seaborn as sns

"""
Plot spike rates in response to different stimuli

Different plotting functions plot different stimuli, or different comparisons
"""

def plot_individual_odor_responses():
    """Plot responses to each odor, presented individually"""
    try:
        os.mkdir(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\individualshape_responses\\')
    except:
        print('dir already made')

    individ_shapes = ['A','B','C']
    for unit in ev_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = ev_sr_df[ev_sr_df['Unit #'] == unit]
        for si, shape in enumerate(individ_shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.plot(np.arange(segment[0], segment[1], .001), curro_df['event spike rate'].mean(), label=shape)

            plt.axvline(0)
        plt.legend()
        plt.title(unit)
        plt.savefig(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\individualshape_responses\\' + str(unit))
        plt.close('all')

def plot_shapepair_responses():
    """Plot responses to each shape pair"""
    try:
        os.mkdir(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\shapepair_responses\\')
    except:
        print('dir already made')

    shapes = ['AB prehab test','BC prehab test','AC prehab test']
    for unit in ev_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = ev_sr_df[ev_sr_df['Unit #'] == unit]
        for si, shape in enumerate(shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.plot(np.arange(segment[0], segment[1], .001), curro_df['event spike rate'].mean(), label=shape)

            plt.axvline(0)
        plt.legend()
        plt.title(unit)
        plt.savefig(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\shapepair_responses\\' + str(unit))
        plt.close('all')

def plot_shapepair_vs_singleshape():
    try:
        os.mkdir(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\shapepair_vs_singleshape\\')
    except:
        print('dir already made')

    shapes = ['A','B','C','AB prehab test','BC prehab test','AC prehab test']
    for unit in ev_sr_df['Unit #'].unique():
        plt.figure(figsize=(18, 10))
        curr_df = ev_sr_df[ev_sr_df['Unit #'] == unit]
        for si, shape in enumerate(shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.plot(np.arange(segment[0], segment[1], .001), curro_df['event spike rate'].mean(), label=shape)

            plt.axvline(0)
        plt.legend()
        plt.title(unit)
        plt.savefig(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\shapepair_vs_singleshape\\' + str(unit))
        plt.close('all')

def plot_posthab_comps():
    try:
        os.mkdir(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\posthab_comps\\')
    except:
        print('dir already made')

    shapes = ['A','B','C','AB posthab test','AC posthab test']
    responsive_units = [19, 26, 59, 114]
    for unit in responsive_units:
        plt.figure(figsize=(18, 10))
        curr_df = ev_sr_df[ev_sr_df['Unit #'] == unit]
        for si, shape in enumerate(shapes):
            curro_df = curr_df[curr_df['substage'] == shape]
            plt.plot(np.arange(segment[0], segment[1], .001), curro_df['event spike rate'].mean(), label=shape)

            plt.axvline(0)
        plt.legend()
        plt.title(unit)
        plt.savefig(ffolder + r'\Figures\spikerates\\' + rec_fname + '\\posthab_comps\\' + str(unit))
        plt.close('all')

get_exp_data = False
segment = [-1, 1]
ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\22-06-20\\'
fnames = ['slice3_merged.h5']
rec_fnames = fnames
get_spikerates = True #set to False to load pre-calculated spikerates


for f in range(0, len(fnames)):
    fname = fnames[f]
    rec_fname = rec_fnames[f]
    ceed_data = ffolder+fname
    Fs=20000
    reader = CeedDataReader(ceed_data)
    # open the data file
    reader.open_h5()

    if get_exp_data:
        from ceed_stimulus import get_all_exps_AB, read_exp_df_from_excel, write_exp_df_to_excel
        exp_df = get_all_exps_AB(ceed_data)
        exp_df.to_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')
        # write_exp_df_to_excel(exp_df, ffolder+'Analysis\\'+fname+'_experiment_df.xlsx', 'Sheet1')
    else:
        exp_df = pd.read_pickle(ffolder+'Analysis\\'+fname+'_exp_df.pkl')

    if get_spikerates:

        """Calculate spike rates for each neuron, segment by events, and organize into dataframe"""
        all_d = []
        # for unit in range(0, len(all_spikes)):
        # Get spiking results from spyking dataframe:
        sk_df = pd.read_pickle(ffolder+'Analysis\\spyking-circus\\' + rec_fname.replace('.h5','')+'.pkl')
        sk_df_csv = pd.read_csv(ffolder+'Analysis\\spyking-circus\\' + rec_fname.replace('.h5','')+'.csv') #remove
        for i, row in sk_df.iterrows():
            if not row['Group'] == 'noise':
                #unitST = all_spikes[unit] / Fs
                unitST = row['Data']# / Fs
                neo_st = []
                tstop = exp_df['t_start'].iloc[-1]*s+10*s
                if unitST[-1] > tstop:
                    tstop = unitST[-1] + 2
                neo_st = neo.core.SpikeTrain(unitST, units=s, t_start=0 * s,
                                                  t_stop=tstop)
                sp_smooth = elephant.statistics.instantaneous_rate(neo_st, sampling_period=1 * pq.ms,
                                                       kernel=elephant.kernels.GaussianKernel(10 * pq.ms))

                for row_i in range(0,exp_df.shape[0]):
                    start = exp_df.iloc[row_i]['t_start']*s+segment[0]*s
                    end = exp_df.iloc[row_i]['t_start']*s+segment[1]*s
                    try:
                        seg_sr = sp_smooth.time_slice(start, (end - .001*s) + .001 * pq.s)
                        seg_st = neo_st.time_slice(exp_df.iloc[row_i]['t_start']*s, exp_df.iloc[row_i]['t_start']*s+.5*s + .001 * pq.s)
                        d = { #Collect relevant info (some of things you may have to remove, or add depending on experiment)
                            'event spike rate': np.squeeze(seg_sr),
                            'evoked n_spikes':len(seg_st),
                            'Unit #': row['ID'],
                            't_start':exp_df.iloc[row_i]['t_start'],
                            'substage': exp_df.iloc[row_i]['substage'],
                            'loop': exp_df.iloc[row_i]['loop'],
                        }
                        all_d.append(d)
                    except ValueError:
                        print('Stimuli after recording stopped')

        ev_sr_df = pd.DataFrame(all_d)
        try:
            os.mkdir(ffolder+'\\Analysis\spikerates')
        except:
            print('directory already made')

        try:
            os.mkdir(ffolder+'\\Figures\spikerates\\'+fname)
        except:
            print('directory already made')

        ev_sr_df.to_pickle(ffolder+'\Analysis\spikerates\\'+fname)
    else:
        ev_sr_df = pd.read_pickle(ffolder+'\Analysis\spikerates\\'+fname)


plot_posthab_comps()




