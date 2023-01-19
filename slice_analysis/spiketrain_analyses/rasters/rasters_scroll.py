import numpy as np
import matplotlib.pyplot as plt
from ceed.analysis import CeedDataReader
import pandas as pd

"""Repeatedly plot rasters from the beginning to the end of the recording"""

ffolder = r''
fname = '_merged.h5'
rec_fname = fname.replace('.h5','')
spyk_f = ffolder+'Analysis\\spyking-circus\\' + rec_fname + '\\' + rec_fname + 'times.result.hdf5'
sk_df = pd.read_pickle(ffolder + 'Analysis\\spyking-circus\\' + rec_fname + '.pkl')

ceed_data = ffolder+fname
Fs=20000
reader = CeedDataReader(ceed_data)
# open the data file
reader.open_h5()
all_d = []

seg_size=10
mint = 0 # in seconds
maxt = 5000 # in seconds
times = np.arange(mint, maxt, seg_size)
units = []

spiketrains = []
for unit in units:
    spiketrains.append(sk_df[sk_df['ID'] == unit].iloc[0]['Data'])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for time in times:
    plt.figure(figsize=(18,1))
    for ui, st in enumerate(spiketrains):
        start = time
        end = time + seg_size
        seg_st = st[np.where(np.logical_and(st>=start, st<end))]
        plt.plot(seg_st,
                 ui * np.ones_like(seg_st), '|', markersize=10, color='blue')
    plt.close()

