import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neo.core import IrregularlySampledSignal
from slice_analysis.spyking_circus_data.spikesorting_results import get_spikes_by_cluster
from quantities import ms, s
from ceed.analysis import CeedDataReader
from ue_analysis import get_surrogate_distribution, ue_window
from slice_analysis.ceed_stimulus_todf import read_exp_df_from_excel, get_stimulus_signal

raster_colors = ["#52aeff", "#dee655"]


def coincidence_plot_pair(h5_file, st1, st2, exp_df, stim='Odor A, weak', bl_duration=1*s, exp_duration=1*s, **kwargs):
    """
    Returns a list of all binary patterns (sequence of 0s and 1s) in a given range

    Parameters
    ------------
    h5_file: HDF5 file
        Merged HDF5 file, containing information from MCS and Ceed. Only used for plotting the stimulus signal.
    st1, st2: SpikeTrain objects
        Pair of neurons to be visualized
    exp_df: DataFrame
        DataFrame describing the stimulation experiments
    stim: str
        Name of a specific Ceed stage (stimulus experiment) that the experiment DataFrame is filtered for
    bl_duration: time quantity
        Length of baseline period to include prior to each experiment
    exp_duration: time quantity
        Length of stimulus period to include for each experiment
    """

    if stim:
        exp_df = exp_df[exp_df['substage'].isin([stim])]

    sts = [st1, st2]
    exp_sts, c_i, winpos, rate, n_emp_win, n_exp_win, js, pval = \
        ue_window(sts, exp_df, bl_duration, exp_duration, **kwargs)

    plt.figure(figsize=(8, 10))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((10, 1), (1, 0), rowspan=3, sharex=ax0)
    ax2 = plt.subplot2grid((10, 1), (4, 0), rowspan=2, sharex=ax0)
    ax3 = plt.subplot2grid((10, 1), (6, 0), rowspan=2, sharex=ax0)
    ax4 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, sharex=ax0)

    # AXIS 0 - STIMULUS INTENSITY PLOT
    reader = CeedDataReader(h5_file)
    reader.open_h5()
    reader.load_application_data()
    reader.load_mcs_data()

    first_exp = int(exp_df['experiment'].values.tolist()[0])
    first_exp_start = exp_df['t_start'].values.tolist()[0]
    if isinstance(first_exp_start, str):  # might be a string if read from excel/csv file
        first_exp_start = np.float(first_exp_start.strip('s')) * s
    t_start = first_exp_start
    start = (t_start - bl_duration.rescale(s)).rescale(ms)
    stop = (t_start + exp_duration.rescale(s)).rescale(ms)

    # For different stimuli, replace shape parameter with any shape that is part of the stimulus stage in Ceed
    if stim == "Odor A, weak" or stim == "Odor A, strong":
        stim_sig = get_stimulus_signal(reader, exp=first_exp, shape='Odor A, circle 1')
    if stim == "Odor B, weak" or stim == "Odor B, strong":
        stim_sig = get_stimulus_signal(reader, exp=first_exp, shape='Odor B, circle 1')

    stim_sig = stim_sig.time_slice(start, stop)
    stim_times = [x - t_start.rescale(ms) for x in stim_sig.times.rescale(ms)]
    stim_fixed = IrregularlySampledSignal(stim_times, stim_sig, units='percent', time_units='ms')
    ax0.set_title("Light stimulus")
    ax0.plot(stim_fixed.times, stim_fixed, "#2172d8", lw=3)
    ax0.set_yticks([])
    ax0.set_ylabel("Intensity", labelpad=10)

    # AXIS 1 - RASTERS W/ COINCIDENT EVENT BOXES
    ax1.set_title("Spike trains and coincident events")
    for i in range(0, len(exp_df)):
        for j in [0, 1]:
            times = exp_sts[i][j].times
            ax1.scatter(times, len(times) * [0.1 * i + .05], color=raster_colors[j], s=5)
    ax1.set_ylabel("Trial")
    yticks = np.arange(0.1, 0.1 * len(exp_df) + 0.1, 0.1)
    ax1.set_yticks(yticks)
    ytick_labels = range(1, 1 + len(exp_df), 1)
    ytick_labels = [str(t) for t in ytick_labels]
    ax1.set_yticklabels(ytick_labels)
    ax1.grid(True, axis='y')
    # Highlight each coincidence w/ rectangle
    for i, trial_coincidences in enumerate(c_i.values()):
        for c in trial_coincidences:
            binsize = kwargs.get('binsize', 5*ms)
            c = c * binsize - bl_duration.rescale(ms)
            rect1 = matplotlib.patches.Rectangle((c, (0.1 * i) + .025), binsize, .05, linewidth=1,
                                                 edgecolor='k', facecolor='none', zorder=1)
            ax1.add_patch(rect1)

    # AXIS 2 - INSTANTANEOUS FIRING RATES
    ax2.set_title("Spike rates")
    ax2.set_ylabel("Rate (hz)")
    ax2.plot(winpos, rate[0], raster_colors[0], label="Neuron 1")
    ax2.plot(winpos, rate[1], raster_colors[1], label="Neuron 2")
    ax2.legend()

    # AXIS 3 - COINCIDENCE RATES
    ax3.set_title("Coincidence rates")
    ax3.set_ylabel("Rate (hz)")
    ax3.plot(winpos, n_emp_win, "#4888d9", label="Observed")
    ax3.plot(winpos, n_exp_win, 'k', label="Expected")
    ax3.legend()

    # AXIS 4 - STATISTICAL SIGNIFICANCE
    plt.title("Statistical significance (time-resolved)")
    ax4.set_ylim([-1, 5.25])
    ax4.set_yticks([0, 2, 4])
    ax4.set_ylabel("Surprise")
    ax4.set_xlabel("Time (ms)")
    ax4.plot(winpos, js, 'k', label="Surprise")

    plt.subplots_adjust(hspace=1.0)
    for ax in [ax0, ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)
    plt.show()
