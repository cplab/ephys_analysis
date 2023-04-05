import os
from quantities import ms, Hz, uV, s
import numpy as np
from neo.core import AnalogSignal, SpikeTrain, IrregularlySampledSignal
from ceed.analysis import CeedDataReader
from ceed.function import CeedFuncRef
from ceed.stage import CeedStageRef
import pandas as pd
from itertools import compress
from openpyxl import load_workbook
from math import isclose

"""Organize ceed stimulus info into a dataframe"""

def get_stimulus_signal(reader: CeedDataReader, exp, shape="enclosed", led="blue", returnas="percent"):
    """this gets the time course of a given stimulus intensity"""

    if shape=='Shape A':
        shape='Shape-3'
    elif shape=='Shape B':
        shape = 'Shape-2'
    reader.load_application_data()
    reader.load_experiment(exp)
    shape_intensities = np.array(reader.shapes_intensity[shape])
    alignment = np.array(reader.electrode_intensity_alignment)

    if led == "red":
        index = 0
    if led == "green":
        index = 1
    if led == "blue":
        index = 2
    intensity = shape_intensities[:-1, index]

    if returnas == "percent":
        intensity = intensity * 100
    if returnas == "norm":
        max = np.max(intensity)
        intensity = intensity / max

    fs = reader.electrodes_metadata['A4']['sampling_frequency']*Hz  # arbitrary electrode
    period = 1./fs
    times = [x * period for x in alignment]
    times = np.array(times)
    intensity = intensity[0:times.shape[0]]
    if len(times)-1 == len(intensity):
        stimulus = IrregularlySampledSignal(times[:-1], intensity, units='percent', time_units='s')
    else:
        stimulus = IrregularlySampledSignal(times, intensity, units='percent', time_units='s')
    return stimulus

def get_stimulus_intensities(reader, exp, df):
    reader.load_mcs_data()
    signal_overall = get_stimulus_signal(reader, exp, 'Granule cell layer shape', led="blue", returnas="percent")
    for i, row in df.iterrows():
        curr_sig = signal_overall.time_slice(t_start=row['t_start'], t_stop=row['t_start']+5)
        df.loc[i, 'Intensity'] = np.round(max(curr_sig).item())
    return df

def get_noisy_parameter(exp_stage):
    """Get values for randomized variable
        variable: a or duration"""
    x = exp_stage.functions[0]
    if len(x.noisy_parameter_samples.keys())>0:
        return x.noisy_parameter_samples[list(x.noisy_parameter_samples.keys())[0]]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_all_exps_enclosed(file):
    reader = CeedDataReader(file)
    reader.open_h5()
    reader.load_application_data()

    exps = reader.experiments_in_file

    reader.load_mcs_data()
    fs = reader.electrodes_metadata['A4']['sampling_frequency'] * Hz
    period = (1. / fs).rescale(s)
    led = ['red', 'green', 'blue']


    columns = ['experiment', 'stage', 'substage', 'duration', 'intensity', 'frequency', 'color', 't_start', 't_stop',
               'signal']
    stim_df = pd.DataFrame(index=range(1000), columns=columns)

    i = 0
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        if reader.electrode_intensity_alignment is not None:
            alignment = np.array(reader.electrode_intensity_alignment)
            times = [round(x * period, 5) for x in alignment]
            # frame_rate = reader.view_controller.effective_frame_rate
            # data = reader.stage_factory.get_all_shape_values(
            #     frame_rate, reader.experiment_stage_name,
            #     pre_compute=reader.view_controller.pre_compute_stages
            # )
            # print(data)
            # break
            stage_dict = reader.stage_factory.stage_names
            #stage = stage_dict[reader.experiment_stage_name]
            stage = reader.experiment_stage
            substages = stage.stages
            signal_overall = get_stimulus_signal(reader, exp, 'enclosed', led="blue", returnas="percent")
            if len(alignment.shape) != 0:
                exp_timer = round(alignment[0] * period, 5)
            if len(substages) == 0:
                n_loops = len(stage.functions[0].noisy_parameter_samples['f'])
                idx = 0
                sig_i = 0
                for l in range(0, n_loops):
                    if (i != 0) and (len(alignment.shape) != 0):
                        stop_index = times.index(true_t_stop)
                        # exp_timer = times[stop_index]
                    stim_df.loc[i, 'experiment'] = str(exp)
                    stim_df.loc[i, 'stage'] = reader.experiment_stage_name
                    stim_df.loc[i, 'substage'] = "None"
                    if len(stage.shapes) == 0:
                        print("Skipping experiment #" + str(exp) + "; no shapes were found within the stage.")
                        continue
                    patterns = [x.name for x in stage.shapes]
                    stim_df.loc[i, 'pattern'] = tuple(patterns)

                    if type(stage.functions[0]) == CeedFuncRef:
                        functions = [x.func.name for x in stage.functions]
                    else:
                        functions = [x.name for x in stage.functions]

                    color_mask = [stage.color_r, stage.color_g, stage.color_b]
                    colors = list(compress(led, color_mask))
                    stim_df.loc[i, 'color'] = colors


                    if len(alignment.shape) == 0:
                        stim_df.loc[i, 't_start'] = np.nan
                        stim_df.loc[i, 't_stop'] = np.nan
                        stim_df.loc[i, 'signal'] = [np.nan]
                        stim_df.loc[i, 'intensity'] = np.nan
                    else:
                        duration = stage.functions[0].duration*s
                        #duration = duration + (1 * (25 / 2999)) * s  # temp solution
                        stim_df.loc[i, 'duration'] = duration
                        stim_df.loc[i, 'frequency'] = stage.functions[0].noisy_parameter_samples['f'][i]
                        found0 = False
                        while not found0:
                            sig_i+=1
                            if signal_overall[sig_i] == 0:
                                found0=True

                        #t_stop = exp_timer + duration + stage.functions[1].copy_expand_ref().duration*s
                        t_stop = signal_overall.times[sig_i]
                        signal = signal_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                        true_t_stop = round(signal.t_stop, 5)

                        stim_df.loc[i, 't_start'] = signal.t_start
                        stim_df.loc[i, 't_stop'] = exp_timer+duration
                        try:
                            stim_df.loc[i, 'signal'] = signal
                        except:
                            stim_df.loc[i, 'signal'] = [signal]
                        stim_df.loc[i, 'intensity'] = max(signal).item()

                        foundnon0 = False
                        while not foundnon0:
                            if sig_i > signal_overall.shape[0]-1:
                                stim_df.dropna(inplace=True, how='all')
                                return stim_df
                            if not signal_overall[sig_i] == 0:
                                foundnon0=True
                                sig_i-=1
                            else:
                                sig_i += 1
                        exp_timer = signal_overall.times[sig_i]

                    i += 1

            else:
                if len(alignment.shape) != 0:
                    exp_timer = round(alignment[0] * period, 5)
                n_loops = 10
                idx = 0
                for l in range(0, n_loops):
                    print('Loop ' + str(l))
                    for k, sub_stage in enumerate(substages):
                        try:
                            sub_stage[0].name
                        except:
                            sub_stage = sub_stage.copy_expand_ref()
                        if (idx != 0) and (len(alignment.shape) != 0):
                            stop_index = times.index(true_t_stop)
                            exp_timer = times[stop_index]  # +1 to grab one sample after the t_stop value for prior sub_stage.
                        stim_df.loc[idx, 'experiment'] = str(exp)
                        stim_df.loc[idx, 'stage'] = reader.experiment_stage_name
                        try:
                            stim_df.loc[idx, 'substage'] = substages[k].get_state()['ref_name']
                        except:
                            stim_df.loc[idx, 'substage'] = sub_stage.name

                        patterns = [x.name for x in sub_stage.shapes]
                        # stim_df.loc[i + k, 'pattern'] = tuple(patterns)
                        # functions = [x.name for x in sub_stage.functions]
                        # stim_df.loc[i + k, 'function'] = functions


                        color_mask = [sub_stage.color_r, sub_stage.color_g, sub_stage.color_b]
                        colors = list(compress(led, color_mask))
                        stim_df.loc[idx, 'color'] = colors

                        if len(alignment.shape) == 0:
                            stim_df.loc[idx, 't_start'] = np.nan
                            stim_df.loc[idx, 't_stop'] = np.nan
                            stim_df.loc[idx, 'signal'] = [np.nan]
                            stim_df.loc[idx, 'intensity'] = np.nan

                        else:
                            try:
                                duration = sub_stage.functions[0].duration * s
                            except:
                                sub_stage = sub_stage.stages[0]
                                duration = sub_stage.functions[0].duration * s

                            timebase = sub_stage.functions[0].timebase
                            if not (timebase.denominator == 1):
                                duration = duration * (timebase.numerator / timebase.denominator)
                            stim_df.loc[idx, 'duration'] = duration
                            t_stop = exp_timer + duration

                            signal_a = signal_a_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                            true_t_stop = round(signal_a.t_stop, 5)
                            stim_df.loc[idx, 't_start'] = signal.t_start
                            stim_df.loc[idx, 't_stop'] = true_t_stop
                            try:
                                stim_df.loc[idx, 'signal'] = signal
                            except:
                                stim_df.loc[idx, 'signal'] = [signal]
                            stim_df.loc[idx, 'intensity'] = max(signal).item()
                            idx+=1

                i += k + 1
    stim_df.dropna(inplace=True, how='all')
    return stim_df


def write_exp_df_to_excel(exp_df, excel, sheet):
    book = load_workbook(excel)
    writer = pd.ExcelWriter(excel)
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    exp_df.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, header=True, index=True)
    writer.save()


def read_exp_df_from_excel(excel, sheet):
    excel_file = pd.ExcelFile(excel, engine='openpyxl')
    stim_df = excel_file.parse(sheet)
    return stim_df


def zero_runs(a):
    """
    Returns a Nx2 dimensional array, with each row containing the start and stop indices of contiguous zeros in the
    original array, a.
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

#Jesse's script
def divy_exp_series(exp_df, exps, sub_exps=10):

    new_exp_df = exp_df
    for exp in exps:
        if not isinstance(exp, str):
            exp = str(exp)
        sub_exp_df = pd.DataFrame(index=range(sub_exps), columns=exp_df.columns) #['experiment', 't_start', 't_stop', 'pattern', 'intensity', 'color', 'signal']
        sub_exp_df.loc[:, 'pattern'] = exp_df[exp_df['experiment']==exp]["pattern"].values[0] * sub_exps

        signal = exp_df[exp_df['experiment']==exp]["signal"].values
        signal = signal[0] #IrregularlySampledSignal object
        signal_aslist = signal.reshape(signal.shape[0]).tolist()
        zero_periods = zero_runs(signal_aslist)
        if zero_periods.shape[0] == sub_exps:
            zero_periods = np.insert(zero_periods, 0, [-1, 0], axis=0)

        if zero_periods.shape[0] != sub_exps+1:
            raise Exception("sub_exps it not equal to the number of intertrial intervals in the original data"
                            + str(zero_periods.shape[0]), str(zero_periods))

        for sub_exp in range(0, sub_exps):
            start_index = zero_periods[sub_exp, 1]
            stop_index = zero_periods[sub_exp+1, 0]
            sub_signal = signal[start_index:stop_index]

            sub_exp_df.loc[sub_exp, "t_start"] = sub_signal.t_start
            sub_exp_df.loc[sub_exp, "t_stop"] = sub_signal.t_stop
            sub_exp_df.loc[sub_exp, "intensity"] = round(max(sub_signal).item(), 2)
            sub_exp_df.loc[sub_exp, "experiment"] = exp + "-" + str(sub_exp)
            sub_exp_df.loc[sub_exp, "signal"] = sub_signal

        new_exp_df = new_exp_df.append(sub_exp_df)
        new_exp_df = new_exp_df[new_exp_df.experiment != exp]
    return new_exp_df


def get_exp_eventdata(reader, select_dataframe):
    """
    Get experiment dataframe using event data

    select_dataframe:: 'combine' -- picks the experiment with the most events
        or 'biggest' -- returns all the experiments in one dataframe
    """
    dfs = []
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        if reader.electrode_intensity_alignment is not None:
            d = {}
            original_root = reader.stage_factory.stage_names[reader.experiment_stage_name]
            for exp_stage, orig_stage in zip(reader.experiment_stage.get_stages(), original_root.get_stages(True)):
                orig_stage = orig_stage.stage if isinstance(s, CeedStageRef) else orig_stage
                d[exp_stage.ceed_id] = exp_stage, orig_stage.name
                for f in exp_stage.functions:
                    for ff in f.get_funcs():
                        d[ff.ceed_id] = ff, ff.name

            # if 'Whole experiment' in d[0][1] or d[0][1]=='Stage-2' or d[0][1]=='Stage-3' or d[0][1]=='GridAll-1' or d[0][1]=='Overall':
            print('Getting event data for ' + str(d[0][1]))
            flat_events = []
            i=0
            while i < len(reader.event_data):
                ev = reader.event_data[i]
                try:
                    if len(d[ev[1]][0].stages) > -1:
                        flat_events.append([ev[0], ev[1], d[ev[1]][1], ev[2]])
                    i+=1
                except:
                    i+=1

            flat_events = [[ev[0], ev[1], d[ev[1]][1], ev[2]] + list(ev[3]) for ev in reader.event_data]

            loop_events = [line for line in flat_events if line[3] == 'start_loop']
            df = pd.DataFrame(loop_events, columns=['Frame','Event','substage','start_loop','sub_loop','loop','time_est'])

            """Drop subevents of substages with multiple stages"""
            evts2drop = ['']
            to_drop = []
            for index, row in df.iterrows():
                if row[2] in evts2drop:
                    to_drop.append(index)
                    # to_drop.append(index+2)
            df = df.drop(labels=to_drop, axis=0)

            df = df.drop(labels=['start_loop'], axis=1)
            df = df.sort_values('Frame', axis=0, ascending=True)
            aligned_times = []
            for index, row in df.iterrows():
                try:
                    aligned_times.append(reader.electrode_intensity_alignment_gpu_rate[row[0]] / 20000)
                except:
                    aligned_times.append('out of range')
            df['t_start'] = aligned_times
            # if df.shape[0] > 1:
            #     df = extract_habituation_evts(df)
            #     df = df.sort_values('t_start', axis=0, ascending=True)
            #     dfs.append(df)
            # df = get_stimulus_intensities(reader, exp, df)

            for i, row in df.iterrows():
                df.at[i, 'Experiment'] = d[0][1]

            dfs.append(df)

    """
    Deal with multiple experiments (can be from running the same experiment under different drugs, or just from running the same experiment multiple times)
    """
    if select_dataframe == 'biggest':
        max_length = 0
        for df in dfs:
            df['drug'] = 'baseline'
            if df.shape[0] > max_length:
                max_length = df.shape[0]
                biggest_df = df
        return biggest_df
    elif select_dataframe == 'combine':
        for di, df in enumerate(dfs):
            if di == 0:
                overall_df = df
            else:
                overall_df = overall_df.append(df)
        return overall_df


def fix_h5_alignment_intensity_series(h5_file, exp_df, drop_exps=None, wait_time=2*s, fs=20000*Hz):
    """
    Function that attempts to align CEED data with MCS data from the merged h5 file when the merge script fails to
    align properly based on periods of zero output from the projector.

    This works frequently, but not reliably, so the user is advised to look at the results to verify
    that the data alignment is correct.


    Parameters
    ------------
    h5_file: h5 file
        An h5 file that has been merged from the MCS and Ceed h5 files, and contains the electrode data
    exp_df: DataFrame
        Containing the Ceed stimulus information
    drop_exps: list
        Experiments to drop from the resulting DataFrame
    resample: bool, default=True
        If True, the data will be resampled at 512*Hz, default=True
    wait_time: time quantity
        The time to wait before considering projector output to be a new stimulus; should be greater than any period
        of zero-output during the stimulus, but less than the inter-stimulus interval
    fs: frequency quantity
        Sampling rate of the dataset
    """
    period = (1/fs).rescale(s).item()
    wait_time = wait_time.rescale(s).item()

    f = h5py.File(h5_file, 'r')
    dig_io = f["data"]["mcs_data"]["data_arrays"]["digital_io"]["data"].value

    stim = np.where(dig_io != 0)[0]  # When (in samples) the projector is projecting some light pattern

    stim_delta_samples = np.diff(stim)
    non_contig = np.where(stim_delta_samples > 1)  # When (in samples) the projected pattern changes, by index of stim/stim_delta

    stim = stim[non_contig].tolist()
    stim_times = [x * period for x in stim]
    stim_times = np.array(stim_times)
    stim_delta_times = np.diff(stim_times)
    deltas_over_wait = np.where(stim_delta_times > wait_time)
    times_over_wait = stim_times[deltas_over_wait]

    if drop_exps is not None:
        for exp in drop_exps:
            exp_df = exp_df.drop(exp)
    exp_df['t_start'] = [t * s for t in times_over_wait]
    exp_df['t_stop'] = [(t + 5) * s for t in times_over_wait]

    return exp_df


if __name__ == "__main__":

    import pprint
    ffolder = r'' #folder containing each slice's merged ceed file (indicated by _merged)
    for fname in os.listdir(ffolder):
         if '_merged.h5' in fname:
            ceed_data = ffolder + fname
            Fs = 20000
            reader = CeedDataReader(ceed_data)
            # open the data file
            reader.open_h5()
            # exp_df = get_exp_eventdata_gridprobe(reader, select_dataframe='combine')  # Grab all experiment information
            exp_df = get_exp_eventdata(reader, select_dataframe='combine')  # Grab all experiment information
            # exp_df = get_exp_eventdata_gridstim()
            exp_df.to_pickle(ffolder + 'Analysis\\' + fname + '_exp_df.pkl') #Make sure you have an Analysis folder here
