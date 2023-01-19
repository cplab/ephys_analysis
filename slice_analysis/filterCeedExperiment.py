from ceed.analysis import CeedDataReader
from filters import butter_lowpass_filter, butter_highpass_filter, butter_bp_filter, iir_notch
from scipy import signal
import numpy as np
from quantities import Hz
import os
import h5py

ffolder = r''
read_merged = True
try:
    os.mkdir(ffolder+'filt_and_rsamp')
except:
    print('Directory already created')
for fname in os.listdir(ffolder):
    if '_merged.h5' in fname:
        ceed_data = ffolder+fname

        if read_merged:
        # create instance that can load the data
            reader = CeedDataReader(ceed_data)
            print('Created reader for file {}'.format(reader.filename))
            # open the data file
            reader.open_h5()
            # load the mcs data into memory

            reader.load_mcs_data()
        else: #read h5
            #TODO: unfinished
            with h5py.File(ffolder+fname, "r") as f:
                a_group_key = list(f.keys())[0]
                dataset = f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData']

        electrodes = sorted(reader.electrodes_data.keys())

        # #Filter and resample data
        rfs = 1000
        Fs = 20000

        sig_length = float(reader.electrodes_data['A4'].shape[0] / Fs)  # get signal length in seconds
        num_samples = int(sig_length * rfs)
        for elec in electrodes:
            reader.electrodes_data[elec] = butter_lowpass_filter(reader.electrodes_data[elec], Fs, 200)
            reader.electrodes_data[elec] = signal.resample(reader.electrodes_data[elec], num_samples)
            reader.electrodes_data[elec] = butter_highpass_filter(reader.electrodes_data[elec], rfs, 1)
            reader.electrodes_data[elec] = iir_notch(reader.electrodes_data[elec], rfs * Hz, frequency=60 * Hz, quality=30, axis=-1)

        np.save(ffolder+'filt_and_rsamp\\' + fname+'.npy',reader.electrodes_data)
