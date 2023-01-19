import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import quantities as pq
import elephant
import neo

"""Plot spike rates over time"""

segment = [-1, 2]
ffolder = r''
fname = '_merged.h5'
rec_fname = fname.replace('.h5','')
sk_df = pd.read_pickle(ffolder + 'Analysis\\spyking-circus\\' + rec_fname + '.pkl')

try:
    os.mkdir(ffolder+'Figures\\rates\\')
except:
    print('folder already made')

try:
    os.mkdir(ffolder+'Figures\\rates\\'+rec_fname)
except:
    print('folder already made')

drug_log = pd.read_excel(ffolder+'Analysis\\drug log.xlsx', sheet_name=rec_fname.replace('_merged',''))
exp_df = pd.read_pickle(ffolder + 'Analysis\\' + fname + '_exp_df.pkl')

def segment():
    """Bin spikes into time segments and count (quicker but less detail)"""
    df_made = False
    for unit in sk_df['ID'].unique():
        if not sk_df[sk_df['ID']==unit].iloc[0]['Group']=='noise':
            st = sk_df[sk_df['ID']==unit]['Data']
            st = st.iloc[0]

            segment_size = 2

            times = np.arange(0, np.max(st), segment_size)
            spikecounts = []
            for time in times:
                spikecounts.append({'Time': time,
                                        'Count': len(st[np.where(np.logical_and(st > time, st < (time + segment_size)))])})

            seg_df = pd.DataFrame(spikecounts)
            if df_made:
                overall_df = overall_df.append(seg_df)
            else:
                df_made = True
                overall_df = seg_df
            plt.figure(figsize=(18,10))
            sns.lineplot(data=seg_df, x='Time', y='Count')
            ax = plt.gca()

            for i, row in drug_log.iterrows():
                time = row['Time - minutes']*60 + row['Time - seconds']
                plt.axvline(time)
                ax.text(time, 0, row['Event'], ha="left", va="top", rotation=0, size=8)
            for drug_exp in exp_df['drug'].unique():
                drug_df = exp_df[exp_df['drug']==drug_exp]
                plt.axvline(drug_df.iloc[0]['t_start'])
                ax.text(drug_df.iloc[0]['t_start'], 0, 'exp', ha="left", va="top", rotation=0, size=8)

            if sk_df[sk_df['ID']==unit].iloc[0]['Group'] == 'good':
                plt.savefig(ffolder+r'Figures\rates\\'+rec_fname+'\\'+str(unit)+'_good.png')
            else:
                plt.savefig(ffolder + r'Figures\rates\\' + rec_fname + '\\' + str(unit) + '.png')
            plt.close()

    sns.lineplot(data=overall_df, x='Time', y='Count')
    ax = plt.gca()

    for i, row in drug_log.iterrows():
        time = row['Time - minutes'] * 60 + row['Time - seconds']
        plt.axvline(time)
        ax.text(time, 0, row['Event'], ha="left", va="top", rotation=0, size=8)

    plt.savefig(ffolder+'Figures\\rates\\'+rec_fname+'_overall.png')
    """Get spikecounts by which event they are from"""
    # seg_spikecounts = []
    # total_spikecounts = []
    # for e_i, evt in enumerate(events):
    #     t_range = [evt['Time'], evt['Time']+5*60]
    #     segments = np.arange(t_range[0], t_range[1], segment_size)
    #     total_spikecounts.append({'Solution':evt['Event'], 'Time':t_range[0], 'Count':len(st[np.where(np.logical_and(st > t_range[0], st < t_range[1]))])})
    #     for seg in segments:
    #         seg_spikecounts.append({'Solution':evt['Event'], 'Time':seg, 'Count':len(st[np.where(np.logical_and(st > seg, st < (seg+segment_size)))])})
    #
    # seg_df = pd.DataFrame(seg_spikecounts)
    #
    # sns.barplot(data=seg_df, x='Solution', y='Count')

def kernel():
    """Use a kernel to estimate spike rate"""
    overall_unit = []
    for unit in sk_df['ID'].unique():
        if not sk_df[sk_df['ID']==unit].iloc[0]['Group']=='noise':
            unitST = sk_df[sk_df['ID']==unit]['Data']
            unitST = unitST.iloc[0]
            neo_st = []
            exp_tstop = exp_df['t_start'].iloc[-1] * pq.s + 50 * pq.s

            # if unitST[-1] > exp_tstop:
            #     tstop = (unitST[-1] + 2) * pq.s
            # else:
            #     tstop = exp_tstop
            tstop = exp_tstop
            neo_st = neo.core.SpikeTrain(unitST, units=pq.s, t_start=0 * pq.s,
                                              t_stop=tstop)
            # neo_st = neo_st.time_slice(0, exp_tstop)
            sp_smooth = elephant.statistics.instantaneous_rate(neo_st, sampling_period=1 * pq.ms,
                                                               kernel=elephant.kernels.AlphaKernel(1000 * pq.ms), center_kernel=False)
            plt.figure(figsize=(18,10))
            plt.plot(np.arange(0, tstop/pq.s, .001)[50000:sp_smooth.shape[0]-50000], np.squeeze(sp_smooth)[50000:sp_smooth.shape[0]-50000])
            ax = plt.gca()
            for i, row in drug_log.iterrows():
                time = row['Time - minutes'] * 60 + row['Time - seconds']
                plt.axvline(time)
                ax.text(time, np.max(sp_smooth), row['Event'], ha="left", va="top", rotation=0, size=8)
            if sk_df[sk_df['ID']==unit].iloc[0]['Group'] == 'good':
                plt.savefig(ffolder+r'Figures\rates\\'+rec_fname+'\\'+str(unit)+'_kernel_good.png')
                overall_unit.append(sp_smooth)
            else:
                plt.savefig(ffolder + r'Figures\rates\\' + rec_fname + '\\' + str(unit) + '_kernel.png')
            plt.close()


    """Plot overall unit"""
    plt.figure(figsize=(18,10))
    overall_mean = np.mean(np.asarray(overall_unit),0)
    plt.plot(np.arange(0, tstop / pq.s, .001)[50000:overall_mean.shape[0] - 50000],
             np.squeeze(overall_mean)[50000:overall_mean.shape[0] - 50000])
    ax = plt.gca()
    for i, row in drug_log.iterrows():
        time = row['Time - minutes'] * 60 + row['Time - seconds']
        plt.axvline(time)
        ax.text(time, np.max(overall_mean), row['Event'], ha="left", va="top", rotation=0, size=8)
    if sk_df[sk_df['ID'] == unit].iloc[0]['Group'] == 'good':
        plt.savefig(ffolder + r'Figures\rates\\' + rec_fname + '\\overall_kernel_good.png')
    else:
        plt.savefig(ffolder + r'Figures\rates\\' + rec_fname + '\\overall_kernel.png')
    plt.close()

if "___main___":
    kernel()