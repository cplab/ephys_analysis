
"""Grid stim scripts"""
def extract_gridstim_events(reader):
    fs = 20000 * Hz
    period = (1. / fs).rescale(s)
    led = ['red', 'green', 'blue']
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        if reader.electrode_intensity_alignment_gpu_rate is not None:
            d = {}
            original_root = reader.stage_factory.stage_names[reader.experiment_stage_name]
            for exp_stage, orig_stage in zip(reader.experiment_stage.get_stages(), original_root.get_stages(True)):
                orig_stage = orig_stage.stage if isinstance(s, CeedStageRef) else orig_stage
                d[exp_stage.ceed_id] = exp_stage, orig_stage.name
                for f in exp_stage.functions:
                    for ff in f.get_funcs():
                        d[ff.ceed_id] = ff, ff.name

            if d[0][1] == 'GridAll' or d[0][1] == 'GridAll-1' or d[0][1]=='GridAll-2':

                alignment = np.array(reader.electrode_intensity_alignment_gpu_rate)
                times = [round(x * period, 5) for x in alignment]

                example_stim = reader.shapes_intensity_rendered_gpu_rate['coord0,0'][:,2]
                #Find where stim starts not being 0, then find its apex, then mark surrounding frames as stim
                # while example_stim[frame] == 0 and frame <= len(example_stim):
                #     frame += 1
                max_frames = []
                intensities = []
                fi = 0
                xs = np.arange(0, 1920, 100)
                ys = np.arange(0, 1080, 100)
                while fi < len(example_stim):
                # for fi in range(0,len(example_stim)):
                    max_intensity = 0
                    if example_stim[fi] != 0:
                        while example_stim[fi] != 0:
                            if example_stim[fi] > max_intensity:
                                max_intensity = example_stim[fi]
                                max_frame = fi
                            fi+=1
                        maxed_out = False
                        for xi in range(0, len(xs)):
                            for yi in range(0, len(ys)):
                                if reader.shapes_intensity_rendered_gpu_rate[f'coord{xs[xi]},{ys[yi]}'][:, 2][max_frame] == 1:
                                    maxed_out = True
                        if not maxed_out:
                            max_frames.append(max_frame)
                            intensities.append(max_intensity)
                    fi+=1

                print('Now, max frames should be stimulation peak frames')
                exp_dicts = []
                for mf in max_frames:
                    exp_dicts.append({'Frame max':mf, 't_max':times[mf], 't_start':times[mf-30]})
                exp_df = pd.DataFrame(exp_dicts)

                # """Why are intensities so often 1???"""
                # import matplotlib.pyplot as plt
                # plt.figure()
                # for i, row in exp_df.iterrows():
                #     print(example_stim[row['Frame max']-30:row['Frame max']+30])
                #     plt.plot(example_stim[row['Frame max']-30:row['Frame max']+30])

                #Make arrays of the max light intensities for each odor, and each square. Divide into which concentration they likely are
                xs = np.arange(0, 1920, 100)
                ys = np.arange(0, 1080, 100)
                intensity_arr = np.empty((len(intensities), len(xs), len(ys)))
                intensity_arr[:] = np.nan
                for i, row in exp_df.iterrows():
                    for xi in range(0,len(xs)):
                        for yi in range(0,len(ys)):
                            intensity_arr[i, xi, yi] = reader.shapes_intensity_rendered_gpu_rate[f'coord{xs[xi]},{ys[yi]}'][:, 2][row['Frame max']]

                odor_sums = []
                for odor in range(0, intensity_arr.shape[0]):
                    odor_sums.append(np.sum(intensity_arr[odor, :, :]))

                #~150 = 1; ~80 = .5; ~40 = .25; ~16 = .1
                threshold = .01
                unique_arrs = []
                uarr_counts = np.zeros((100))
                odor_labels = np.empty(intensity_arr.shape[0])
                odor_labels[:] = np.nan
                for oi in range(0,intensity_arr.shape[0]):
                    odor_arr = intensity_arr[oi, :, :]
                    any_close = False
                    if np.sum(odor_arr) > 100:
                        continue
                    for ui, unique_arr in enumerate(unique_arrs):
                        perc_diff = 0
                        for xi in range(0, len(xs)):
                            for yi in range(0, len(ys)):
                                perc_diff += np.abs((unique_arr[xi, yi] - odor_arr[xi, yi]) / odor_arr[xi, yi])
                        max_total = np.max([np.sum(odor_arr), np.sum(unique_arr)])
                        percent_diff = perc_diff / (len(xs)*len(ys))
                        print(percent_diff)
                        if percent_diff < threshold:
                            if any_close:
                                print('This odor is close to 2 other odors!')
                            uarr_counts[ui] += 1
                            odor_labels[oi] = ui
                            any_close = True
                    if not any_close:
                        unique_arrs.append(odor_arr)
                        odor_labels[oi] = len(unique_arrs)-1 #label will be the next entry
                        uarr_counts[len(unique_arrs)-1] += 1

                print(str(len(unique_arrs))+ ' unique sums')

                pairs = []
                ratios = [5,2.4,  2, .2, .4, .5]
                for ui, unique_arr in enumerate(unique_arrs):
                    for ui_other, unique_arr_other in enumerate(unique_arrs):
                        ratio = np.round(unique_arr[0,0] / unique_arr_other[0,0], 3)
                        which_ratio = np.where(ratio==ratios)
                        if which_ratio[0].shape[0] > 0:
                            pairs.append({'Pair':[ui, ui_other], 'Ratio':ratio})

                print('should hit a breakpoint and now define odor groups based off the pairs IDd')
                odor_groups = [[11,3,0], [7,5,1], [8,2,6], [10,9,4]] #ordered by concentration, made manually using the pairs
                labels = []
                odors = []
                concentrations = []
                concs = [.1, .25, .5]
                for i, row in exp_df.iterrows():
                    if np.isnan(odor_labels[i]):
                        odors.append(np.nan)
                        concentrations.append(np.nan)
                        labels.append(odor_labels[i])
                        continue
                    labels.append(odor_labels[i])
                    for gi, group in enumerate(odor_groups):
                        for ci, odor in enumerate(group):
                            if odor_labels[i] == odor:
                                odors.append(gi)
                                concentrations.append(concs[ci])
                exp_df['Concentration'] = concentrations
                exp_df['Odor'] = odors

    return exp_df

def get_exp_eventdata_gridstim(reader, select_dataframe):
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

            if d[0][1]=='GridEveryOther-1' or d[0][1]=='GridEveryOther-2':
                print('Getting event data for ' + str(d[0][1]))
                flat_events = []
                i=0
                while i < len(reader.event_data):
                    ev = reader.event_data[i]
                    try:
                        if len(d[ev[1]][0].stages) > -1:
                            flat_events.append([ev[0], ev[1], ev[3][0], ev[3][1], ev[3][2], ev[3][3], ev[3][4]])
                        i+=1
                    except:
                        i+=1

                # flat_events = [[ev[0], ev[1], d[ev[1]][1], ev[2]] + list(ev[3]) for ev in reader.event_data]

                df = pd.DataFrame(flat_events, columns=['Frame','ceed ID','Trial num','Odor','original intensity','presented intensity','conc'])
                df = df.drop('conc', axis=1)
                df = df.drop([0])
                """Right now, concs is wrong. Fix"""
                correct_concentrations = []
                for i, row in df.iterrows():
                    conc1 = np.round(reader.shapes_intensity_rendered_gpu_rate[f'coord{0},{0}'][:, 2][int(row['Frame']+30)],2)
                    conc2 = np.round(reader.shapes_intensity_rendered_gpu_rate[f'coord{200},{0}'][:, 2][int(row['Frame'] + 30)],2)
                    if conc1 > 0:
                        correct_concentrations.append(np.round(conc1,2))
                    if conc2 > 0:
                        correct_concentrations.append(np.round(conc2,2))
                df['Conc'] = correct_concentrations


                """Drop subevents of substages with multiple stages"""
                # larger_events = ['A weak B medium cos', 'A weak B medium sq','A strong B medium cos', 'A strong B medium sq',
                #                  'A medium B weak cos', 'A medium B weak sq', 'A medium B strong cos', 'A medium B strong sq',
                #                  'A strong B weak sq', 'A weak B strong sq', 'A strong B weak ramp', 'A weak B strong ramp',
                #                  'A delay B', 'B delay A']
                # larger_events = ['Whole experiment']
                # to_drop = []
                # for index, row in df.iterrows():
                #     if row[2] in larger_events:
                #         to_drop.append(index)
                #         # to_drop.append(index+2)

                # df = df.drop(labels=to_drop, axis=0)
                # df = df.drop(labels=['start_loop'], axis=1)
                df = df.sort_values('Frame', axis=0, ascending=True)
                aligned_times = []
                for index, row in df.iterrows():
                    try:
                        aligned_times.append(reader.electrode_intensity_alignment_gpu_rate[int(row['Frame'])] / 20000)
                    except:
                        aligned_times.append('out of range')
                df['t_start'] = aligned_times
                # if df.shape[0] > 1:
                #     df = extract_habituation_evts(df)
                #     df = df.sort_values('t_start', axis=0, ascending=True)
                #     dfs.append(df)
                # df = get_stimulus_intensities(reader, exp, df)
                dfs.append(df)

    # stage_names = [s.stage.name if isinstance(s, CeedStageRef) else s.name for s in
    #                original_root.stages]

    # #get stage name using stage ID
    # stage_index = stage_names.index('B strong sq')
    # ceed_id = reader.experiment_stage.stages[stage_index].ceed_id
    #
    # #filter for strong, find when first event started
    # strong = df[df[2] == 'B strong sq']
    # strong_loop_0 = strong[strong[3] == 'loop_start']
    # strong = strong[strong[5] == 0]
    # i = strong.iloc[0][0]
    if select_dataframe == 'biggest':
        max_length = 0
        for df in dfs:
            if df.shape[0] > max_length:
                max_length = df.shape[0]
                biggest_df = df
        return biggest_df
    elif select_dataframe == 'combine':
        print('adding str Carb to all events in second experiment')
        for di, df in enumerate(dfs):
            if di == 0:
                df['substage'] = '0 uM Carb baseline ' + df['substage']
                overall_df = df
            elif di == 1:
                df['substage'] = '50 uM Carb ' + df['substage']
                overall_df = overall_df.append(df)
            elif di == 3:
                df['substage'] = '100 uM Carb ' + df['substage']
                overall_df = overall_df.append(df)
        return overall_df


def get_exp_eventdata_gridprobe(reader, select_dataframe):
    """
    Get experiment dataframe using event data

    select_dataframe:: 'combine' -- picks the experiment with the most events
        or 'biggest' -- returns all the experiments in one dataframe
    """
    dfs = []
    for j, exp in enumerate(reader.experiments_in_file):
        reader.load_experiment(exp)
        # if reader.electrode_intensity_alignment is not None:
        d = {}
        original_root = reader.stage_factory.stage_names[reader.experiment_stage_name]
        for exp_stage, orig_stage in zip(reader.experiment_stage.get_stages(), original_root.get_stages(True)):
            orig_stage = orig_stage.stage if isinstance(s, CeedStageRef) else orig_stage
            d[exp_stage.ceed_id] = exp_stage, orig_stage.name
            for f in exp_stage.functions:
                for ff in f.get_funcs():
                    d[ff.ceed_id] = ff, ff.name

        if d[0][1]=='GridSubset-1' or d[0][1]=='GridSubset-2':
            print('Getting event data for ' + str(d[0][1]))
            flat_events = []
            i=0
            while i < len(reader.event_data):
                ev = reader.event_data[i]
                try:
                    if len(d[ev[1]][0].stages) > -1:
                        flat_events.append([ev[0], ev[1], ev[2], ev[3][0]])
                    i+=1
                except:
                    i+=1

            df = pd.DataFrame(flat_events, columns=['Frame','ceed ID','Trial num','ShapeOn'])

            df = df.sort_values('Frame', axis=0, ascending=True)
            aligned_times = []
            for index, row in df.iterrows():
                try:
                    aligned_times.append(reader.electrode_intensity_alignment_gpu_rate[int(row['Frame'])] / 20000)
                except:
                    aligned_times.append('out of range')
            df['t_start'] = aligned_times
            df = remove_duplicate_events(df)
            dfs.append(df)

    if select_dataframe == 'biggest':
        max_length = 0
        for df in dfs:
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

def get_exp_eventdata_grid_similarity_stim(reader, select_dataframe, hab=True):
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

            if 'GridAll' in d[0][1] or 'GridAllHab' in d[0][1] or 'GridAllHabSim' in d[0][1]:
                print('Getting event data for ' + str(d[0][1]))
                flat_events = []
                i=0
                while i < len(reader.event_data):
                    ev = reader.event_data[i]
                    try:
                        if len(d[ev[1]][0].stages) > -1:
                            if hab:
                                flat_events.append([ev[0], ev[1], ev[3][0], ev[3][1], ev[3][2], ev[3][3], ev[3][4], ev[3][5]*2])
                            else:
                                flat_events.append([ev[0], ev[1], ev[3][0], ev[3][1], ev[3][2], ev[3][3] * 2])
                        i+=1
                    except:
                        i+=1

                # flat_events = [[ev[0], ev[1], d[ev[1]][1], ev[2]] + list(ev[3]) for ev in reader.event_data]
                if hab:
                    df = pd.DataFrame(flat_events, columns=['Frame','ceed ID','Trial num','Hab trial num','Dishab ITI','Distance','Initial odor intensity','Presented odor intensity'])
                else:
                    df = pd.DataFrame(flat_events, columns=['Frame','ceed ID','Trial num','Distance','Initial odor intensity','Presented odor intensity'])
                df = df.sort_values('Frame', axis=0, ascending=True)
                aligned_times = []
                for index, row in df.iterrows():
                    try:
                        aligned_times.append(reader.electrode_intensity_alignment_gpu_rate[int(row['Frame'])] / 20000)
                    except:
                        aligned_times.append('out of range')
                #align using start time only, and frames ceed presented at
                # print('Not using actual ceed alignment, but approximating')
                # experiment_start = reader.electrode_intensity_alignment_gpu_rate[df.loc[0, 'Frame']] / 20000
                # aligned_times = []
                # for index, row in df.iterrows():
                #     aligned_times.append(row['Frame']/118.76+experiment_start)
                df['t_start'] = aligned_times
                # if df.shape[0] > 1:
                #     df = extract_habituation_evts(df)
                #     df = df.sort_values('t_start', axis=0, ascending=True)
                #     dfs.append(df)
                # df = get_stimulus_intensities(reader, exp, df)
                df = remove_duplicate_events(df)
                dfs.append(df)

    if select_dataframe == 'biggest':
        max_length = 0
        for df in dfs:
            if df.shape[0] > max_length:
                max_length = df.shape[0]
                biggest_df = df
        return biggest_df
    elif select_dataframe == 'combine':
        print('combinding experiments')
        for di, df in enumerate(dfs):
            df['Experiment'] = di
            if di == 0:
                overall_df = df
            else:
                overall_df = overall_df.append(df)
        return overall_df


"""Habituation stim scripts"""
def extract_habituation_evts(df):
    #second event will always have 15s delay, first event will depend on the length of the previous stage
    df_start = df[df['start_loop']=='start_loop']
    df_end = df[df['start_loop']=='end_loop']
    evts = []
    for i, row in df_start.iterrows():
        if i == 0:
            evts.append(
                {'substage': 'ITI infinite s (first event)', 't_start': row['t_start']})
            stage_length = df_end.iloc[i]['t_start'] - row['t_start']
            ITI = round(stage_length - 16, 1)
            evts.append(
                {'substage': 'ITI ' + str(ITI) + 's', 't_start': row['t_start'] + ITI + .5})
            evts.append({'substage': 'ITI 15s',
                         't_start': row['t_start'] + ITI + 16})
        else:
            #get length of stage
            try:
                if not isinstance(df_end.iloc[i]['t_start'], str):  # if it is a string, out of range
                    stage_length = df_end.iloc[i]['t_start'] - row['t_start']
                    ITI = round(stage_length - 16, 1)
                evts.append(
                    {'substage': 'ITI ' + str(ITI) + 's', 't_start': row['t_start']+ITI+.5})
                evts.append({'substage': 'ITI 15s',
                             't_start': row['t_start'] + ITI + 16})  # there will always be an event 15s later
            except:
                print('huh')
        print(i)

    return pd.DataFrame(evts)

def extract_habituation_dishabituation_evts(df):
    """Go through df, and for each Odor+ITIxs+dishabxs, mark all the odor+ITIs by their #, and
    add the dishabituation stim using the ITI"""
    newdicts = []
    in_hab = False
    counter = 1
    for i, row in df.iterrows():
        if 'ITI ' in row['substage']:
            in_hab = False
            dishab_ITI = row['substage'].replace('ITI ','')
            dishab_ITI = float(dishab_ITI.replace('s', ''))
            t_start = row['t_start'] + dishab_ITI
            newdicts.append({'Frame':row['Frame'], 'substage':'Dishab stim',
                             'Counter':np.nan, 'hab_ITI':hab_ITI, 'dishab_ITI':hab_ITI+dishab_ITI, 't_start':t_start})
        if in_hab:
            # if row['substage'][-3]=='t':
            #     hab_ITI = float('.'+row['substage'][-2])
            # else:
            #     hab_ITI = float(row['substage'][-2])
            if not row['t_start'] == newdicts[-1]['t_start']:
                newdicts.append({'Frame':row['Frame'], 'substage':'Hab stim', 'hab ITI':hab_ITI, 'dishab_ITI':np.nan,
                                 'Counter':counter, 't_start':row['t_start']})
                counter += 1
            # if counter==5:
            #     hab_ITI = np.round(row['t_start'] - df.iloc[i-1]['t_start'] - .5,2)


        if 'Odor+ITI' in row['substage'] and 'dishab' in row['substage']:
            in_hab = True
            counter = 1
            #add the first one now, since it can come before this event and therefore get skipped
            if row['substage'][9] == 't':
                hab_ITI = float('.' + row['substage'][10])
            else:
                hab_ITI = float(row['substage'][8])
            endi = row['substage'].find('s+')
            starti = row['substage'].find('ITI')+3
            if row['substage'][starti] == 'p':
                hab_ITI = float('.'+row['substage'][starti+2:endi])
            else:
                hab_ITI = float(row['substage'][starti:endi])
            newdicts.append({'Frame':row['Frame'], 'substage':'Hab stim', 'hab ITI':hab_ITI, 'dishab_ITI':np.nan,
                             'Counter':counter, 't_start':row['t_start']})
            counter+=1
    return pd.DataFrame(newdicts)

"""Other"""
def get_all_exps_AB(file):
    reader = CeedDataReader(file)
    reader.open_h5()
    reader.load_application_data()

    exps = reader.experiments_in_file

    reader.load_mcs_data()
    fs = reader.electrodes_metadata['A4']['sampling_frequency'] * Hz
    period = (1. / fs).rescale(s)
    led = ['red', 'green', 'blue']


    columns = ['experiment', 'stage', 'substage', 'duration', 'intensity A', 'intensity B', 'color', 't_start', 't_stop',
               'signal A', 'signal B']
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
            stage = stage_dict[reader.experiment_stage_name]
            substages = stage.stages
            signal_a_overall = get_stimulus_signal(reader, exp, 'Shape A', led="blue", returnas="percent")
            signal_b_overall = get_stimulus_signal(reader, exp, 'Shape B', led="blue", returnas="percent")
            if len(substages) == 0:
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
                stim_df.loc[i, 'function'] = functions

                color_mask = [stage.color_r, stage.color_g, stage.color_b]
                colors = list(compress(led, color_mask))
                stim_df.loc[i, 'color'] = colors

                if len(alignment.shape) == 0:
                    stim_df.loc[i, 't_start'] = np.nan
                    stim_df.loc[i, 't_stop'] = np.nan
                    stim_df.loc[i, 'signal'] = [np.nan]
                    stim_df.loc[i, 'intensity A'] = np.nan
                    stim_df.loc[i, 'intensity B'] = np.nan
                else:
                    stim_df.loc[i, 't_start'] = times[0]
                    stim_df.loc[i, 't_stop'] = times[-1]
                    # signal = get_stimulus_signal(reader, exp, patterns[0], led=colors[0], returnas="percent")
                    # stim_df.loc[i, 'signal'] = signal
                    stim_df.loc[i, 'intensity A'] = max(signal_a).item()
                    stim_df.loc[i, 'intensity B'] = max(signal_b).item()
                i += 1

            else:
                if len(alignment.shape) != 0:
                    exp_timer = round(alignment[0] * period, 5)
                n_loops = 20
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
                            stim_df.loc[idx, 'intensity A'] = np.nan
                            stim_df.loc[idx, 'intensity B'] = np.nan

                        else:
                            try:
                                duration = sub_stage.functions[0].duration * s
                            except:
                                sub_stage = sub_stage.stages[0]
                                duration = sub_stage.functions[0].duration * s

                            timebase = sub_stage.functions[0].timebase
                            if not (timebase.denominator == 1):
                                duration = duration * (timebase.numerator / timebase.denominator)
                            if 'A delay B' in stim_df['substage'][idx]:
                                duration = duration+(6*(25/2999))*s #temp solution
                            if 'B delay A' in stim_df['substage'][idx]:
                                duration = duration + sub_stage.functions[1].duration * s  # temp solution
                            stim_df.loc[idx, 'duration'] = duration
                            t_stop = exp_timer + duration

                            signal_a = signal_a_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                            signal_b = signal_b_overall.time_slice(t_start=exp_timer, t_stop=t_stop)
                            true_t_stop = round(signal_a.t_stop, 5)
                            stim_df.loc[idx, 't_start'] = signal_a.t_start
                            stim_df.loc[idx, 't_stop'] = true_t_stop
                            try:
                                stim_df.loc[idx, 'signal A'] = signal_a
                                stim_df.loc[idx, 'signal B'] = signal_b
                            except:
                                stim_df.loc[idx, 'signal A'] = [signal_a]
                                stim_df.loc[idx, 'signal B'] = [signal_b]
                            stim_df.loc[idx, 'intensity A'] = max(signal_a).item()
                            stim_df.loc[idx, 'intensity B'] = max(signal_b).item()
                            idx+=1

                i += k + 1
    stim_df.dropna(inplace=True, how='all')
    return stim_df