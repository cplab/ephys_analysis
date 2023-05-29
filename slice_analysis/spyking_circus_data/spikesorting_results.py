import numpy as np
import sys
from neo.core import SpikeTrain
from quantities import Hz, ms, s
from tqdm import tqdm
import pandas as pd
import h5py


all_electrodes = [
    'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
    'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
    'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
    'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
    'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12',
    'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
    'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12',
    'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11',
    'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10',
    'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]


def get_circus_results(results_folder, fs=20000*Hz, get_electrodes=False, get_groups=False):
    """
    Return the spike-sorting results after running the SpykingCircus algorithm

    Parameters
    ------------
    results_folder: str
        The directory where SpykingCircus stored spike-sorting results (should end in .GUI)
    fs: frequency quantity
        Sampling rate
    get_electrodes: bool
        If true, returns the electrode IDs for each cluster
    get_groups: bool
        If true, returns the group each cluster was put into
    """
    gui_base = r"{results_folder}/{file}.{ext}"
    period = (1./fs).rescale(ms)
    times = np.load(gui_base.format(results_folder=results_folder, file="spike_times", ext="npy"))
    clusters = np.load(gui_base.format(results_folder=results_folder, file="spike_clusters", ext="npy"))
    cluster_ids = np.unique(clusters)

    if get_electrodes:
        clusters_h5 = gui_base.split('.')[0] + ".clusters.hdf5"
        clusters_h5 = h5py.File(clusters_h5, 'r')
        electrodes = list(clusters_h5.get('electrodes'))

    circus_df = pd.DataFrame({"ID": cluster_ids, "Electrode": np.nan, "Group":np.nan, "Data": np.nan})
    circus_df = circus_df.astype(object)
    pbar = tqdm(circus_df.iterrows(), total=circus_df.shape[0], file=sys.stdout, desc="Getting SpykingCircus results")

    i = 0
    for index, cluster in pbar:
        id_ = cluster["ID"]
        cluster_indices = np.where(clusters == id_)
        id_st = times[cluster_indices]
        id_st = [period * x for x in id_st]
        id_st = np.array(id_st)
        id_st.reshape(id_st.shape[0], )
        cluster["Data"] = id_st

        if get_electrodes:
            channel_map = np.load(gui_base.format(results_folder=results_folder, file="channel_map", ext="npy"))
            if id_ in range(0, len(electrodes)):
                electrode_index = electrodes[id_]
                cluster["Electrode"] = channel_map[electrode_index]
        i += 1
    pbar.close()

    if get_groups:
        tsv_file = open(gui_base.format(results_folder=results_folder, file="cluster_group", ext="tsv"))
        group_df = pd.read_csv(tsv_file, sep='\t')

        for _, group in group_df.iterrows():
            id = group["cluster_id"]
            group = group["group"]
            circus_df.loc[circus_df['ID'] == id, 'Group'] = group
    return circus_df


def get_spikes_by_cluster(results_folder, cluster_id, fs=20000*Hz, t_stop=3600*s):
    """
    Return the spike-sorting results after running the SpykingCircus algorithm

    Parameters
    ------------
    results_folder: str
        The directory where SpykingCircus stored spike-sorting results (should end in .GUI)
    cluster_id: int
        ID of the cluster to be returned
    fs: frequency quantity
        Sampling rate
    t_stop: recording stop time
        Required to return a SpikeTrain object; if unsure, overestimate
    """
    gui_base = f"{results_folder}/{file}.{ext}"
    period = (1./fs).rescale(ms)
    times = np.load(gui_base.format(results_folder=results_folder, file="spike_times", ext="npy"))
    clusters = np.load(gui_base.format(results_folder=results_folder, file="spike_clusters", ext="npy"))
    cluster_indices = np.where(clusters == cluster_id)
    spikes = times[cluster_indices]
    spikes_converted = [period * x for x in spikes]
    spikes_converted = np.array(spikes_converted)
    spikes_converted.reshape(spikes_converted.shape[0], )
    spikes_converted = SpikeTrain(spikes_converted*ms, t_stop)
    return spikes_converted


def fix_electrode_ids(circus_df):
    electrode_map = {}
    for i in range(120):
        electrode_map[i] = all_electrodes[i]
    new_electrodes = list(map(electrode_map.get, circus_df["Electrode"]))
    fixed_circus_df = circus_df
    fixed_circus_df["Electrode"] = new_electrodes
    print("Electrode IDs fixed!")
    return fixed_circus_df
