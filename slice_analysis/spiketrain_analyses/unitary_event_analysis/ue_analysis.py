import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import neo
from neo.core import (
    IrregularlySampledSignal)
from quantities import uV, Hz, ms, s
import scipy
import pandas as pd
from openpyxl import load_workbook
import elephant
from elephant import unitary_event_analysis as ue
from elephant.spike_train_surrogates import dither_spikes
from tqdm import tqdm
import itertools


def get_binary_patterns(n_neurons, min_pattern_size=2, max_pattern_size=2):
    """
    Returns a list of all binary patterns (sequence of 0s and 1s) in a given range

    Parameters
    ------------
    n_neurons: int
        Number of neurons in the current analysis
    min_pattern_size: int
        Minimum # of neurons active (number of 1s in pattern)
    max_pattern_size: int
        Maximum # of neurons active (number of 1s in pattern)

    Returns
    ------------
    binary_patterns: array of binary patterns
    """
    binary_patterns = np.asarray(list(itertools.product([0, 1], repeat=n_neurons)), dtype=object)
    i = np.where((np.sum(binary_patterns, axis=1) >= min_pattern_size) & (np.sum(binary_patterns, axis=1)
                                                                          <= max_pattern_size))
    binary_patterns = binary_patterns[i, :][0]

    binary_patterns = np.array(binary_patterns).T
    return binary_patterns


def get_experiment_spikes(sts, exp_df, bl_duration=1*s, exp_duration=1*s, binsize=5*ms):
    """
    Returns a list of all binary patterns (sequence of 0s and 1s) in a given range

    Parameters
    ------------
    sts: list of Neo SpikeTrain objects
    exp_df: DataFrame
        DataFrame describing the stimulation experiments
    bl_duration: time quantity
        Length of baseline period to include prior to each experiment
    exp_duration: time quantity
        Length of stimulus period to include for each experiment
    binsize: time quantity
        Time resolution of spike binning

    Returns
    ------------
    exp_sts: list of list of SpikeTrain objects
        SpikeTrains, time sliced for each experiment
        shape: different experiments --> 0-axis
               different neurons --> 1-axis
    bs_array: numpy array
        Binned SpikeTrains, time sliced for each experiment
        shape: different experiments --> 0-axis
               different neurons --> 1-axis
               time bins --> 2-axis
    """
    n_cols = int((bl_duration+exp_duration).rescale(ms) / binsize.rescale(ms))
    bs_array = np.empty([len(exp_df), len(sts), n_cols])
    exp_sts = [0] * len(exp_df)

    for i, (_, experiment) in enumerate(exp_df.iterrows()):
        t_start = experiment['t_start']
        if type(t_start) == str:
            t_start = t_start.strip('s')  # if experiment DataFrame is read from excel, start time will be a string
        t_start = np.float(t_start) * s
        start = (t_start - bl_duration).rescale(ms)
        stop = (t_start + exp_duration).rescale(ms)
        st_list_exp = [0] * len(sts)
        for j, st in enumerate(sts):
            st_cut = [x.rescale(ms).item()-t_start.rescale(ms).item() for x in st if start <= x <= stop]
            st_cut = neo.SpikeTrain(st_cut, units=ms, t_start=-bl_duration.rescale(ms), t_stop=exp_duration.rescale(ms))
            st_list_exp[j] = st_cut
            bs = elephant.conversion.BinnedSpikeTrain(st_cut, t_start=-bl_duration.rescale(ms),
                                                      t_stop=exp_duration.rescale(ms), binsize=binsize)
            bs_array[i, j, :] = bs.to_array()
            exp_sts[i] = st_list_exp
    bs_array[np.where(bs_array > 1)] = 1
    return exp_sts, bs_array


def get_surrogate_distribution(exp_sts, t_start=0*s, t_stop=2*s, binsize=5*ms, winsize=100*ms,
                               winstep=10*ms, n_surr=1000, d=10*ms):
    """
    Returns a list of all binary patterns (sequence of 0s and 1s) in a given range

    Parameters
    ------------
    exp_sts: list of list of Neo SpikeTrain objects
        shape: different experiments --> 0-axis
        different neurons --> 1-axis
    t_start: time quantity
        start time
    t_stop: time quantity
        stop time
    binsize: time quantity
        time resolution of spike binning
    winsize: time quantity
        window over which statistics (rate, surprise value) are calculated
    winstep: time quantity
        window step size
    n_surr: int
        number of surrogates to bootstrap
    d: time quantity
        dither variable; each spike in the surrogate data will be shifted between [-d, d] with respect to original time

    Returns
    ------------
    surrogate_exp: numpy array
        the number of coincidences in each bin across all surrogates
        shape: bins --> 0-axis
               surrogates --> 1-axis
    """
    n_exps = len(exp_sts)
    duration = t_stop.rescale(ms) - t_start.rescale(ms)
    bins = int(duration/binsize.rescale(ms))
    n_window_positions = int(int(duration-winsize.rescale(ms))/int(winstep.rescale(ms)))+1
    winsize_bins = int(winsize.rescale(ms).item()/binsize.rescale(ms).item())
    winstep_bins = int(winstep.rescale(ms).item()/binsize.rescale(ms).item())

    surrogate_exp = np.zeros([n_window_positions, n_surr])
    for surr in tqdm(range(n_surr)):
        sbs_array = np.zeros([n_exps, 2, bins])

        for e in np.arange(0, len(exp_sts)):
            exp_st1, exp_st2 = exp_sts[e][0], exp_sts[e][1]
            st1_surrogate = dither_spikes(exp_st1, dither=d)[0]
            st2_surrogate = dither_spikes(exp_st2, dither=d)[0]
            s_bs1 = elephant.conversion.BinnedSpikeTrain(st1_surrogate, t_start=t_start.rescale(ms),
                                                         t_stop=t_stop.rescale(ms), binsize=binsize)
            s_bs2 = elephant.conversion.BinnedSpikeTrain(st2_surrogate, t_start=t_start.rescale(ms),
                                                         t_stop=t_stop.rescale(ms), binsize=binsize)
            sbs_array[e, 0, :] = s_bs1.to_array()
            sbs_array[e, 1, :] = s_bs2.to_array()
        sbs_array[np.where(sbs_array > 1)] = 1

        for pos in np.arange(0, n_window_positions):
            win_sbs_array = sbs_array[:, :, pos*winstep_bins:pos*winstep_bins+winsize_bins]
            surrogate_emp, _ = ue.n_emp_mat_sum_trial(win_sbs_array, 2, [3])
            surrogate_exp[pos, surr] = surrogate_emp
    return surrogate_exp


def ue_window(sts, exp_df, bl_duration=1*s, exp_duration=1*s, surrogates=True, winsize=50*ms, binsize=5*ms,
              winstep=10*ms, **kwargs):
    """
    Returns a list of all binary patterns (sequence of 0s and 1s) in a given range

    Parameters
    ------------
    sts: list of Neo SpikeTrain objects
        shape: different experiments --> 0-axis
        different neurons --> 1-axis
    exp_df: time quantity
        start time
    bl_duration: time quantity
        Length of baseline period to include prior to each experiment
    exp_duration: time quantity
        Length of stimulus period to include for each experiment
    winsize: time quantity
        window over which statistics (rate, surprise value) are calculated
    binsize: time quantity
        time resolution of spike binning
    winstep: time quantity
        window step size
    surrogates: bool
        If True, calculates surprise value based on bootstrapped surrogates with dithered spike times. If False,
        calculates the surprise value analytically based on spike rates within the window.
    Returns
    ------------
    surrogate_exp: numpy array
        the number of coincidences in each bin across all surrogates
        shape: bins --> 0-axis
               surrogates --> 1-axis
    """
    bl_duration = bl_duration.rescale(ms)
    n_neurons = len(sts)
    patterns = get_binary_patterns(n_neurons)
    pattern_hash = [ue.hash_from_pattern(patterns, n_neurons)]

    exp_sts, bs_array = get_experiment_spikes(sts, exp_df, bl_duration=bl_duration, exp_duration=exp_duration,
                                              binsize=binsize)
    _, _, n_exp, n_emp, indices = ue._UE(bs_array, n_neurons, pattern_hash)

    ue_win = ue.jointJ_window_analysis(exp_sts, binsize, winsize, winstep, pattern_hash)
    js, rate, n_exp_win, n_emp_win, c_i = ue_win["Js"], ue_win["rate_avg"], ue_win["n_exp"], ue_win["n_emp"], \
                                                  ue_win["indices"]
    dist_exp, _ = ue.gen_pval_anal(bs_array, n_neurons, pattern_hash)
    pval = dist_exp(n_emp)

    rate = [x * 1000 for x in rate]  # Convert to Hz
    rate = np.array(rate).T
    winpos = ue._winpos(-bl_duration.rescale(s), exp_duration.rescale(s), winsize, winstep)

    # Overwrite n_exp_win, js, and pval with values from surrogate distribution
    if surrogates:
        surrogate_distribution = get_surrogate_distribution(exp_sts, t_start=-bl_duration, t_stop=exp_duration,
                                                            binsize=binsize, winsize=winsize, winstep=winstep, **kwargs)
        n_exp_win = np.percentile(surrogate_distribution, 50, axis=1)  # median

        js, pval = np.empty(len(winpos)), np.empty(len(winpos))
        for t, (emp, exp) in enumerate(zip(n_emp_win, n_exp_win)):
            surrogate_distribution_win = surrogate_distribution[t]
            p = 1 - scipy.stats.percentileofscore(surrogate_distribution_win, emp) / 100.
            pval[t] = p
            js[t] = ue.jointJ(p)
    return exp_sts, c_i, winpos, rate, n_emp_win, n_exp_win, js, pval
