import os
import pandas
import itertools
import numpy as np
import detect_outliers
from plot_on_mouse import plm
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import defaultdict
from behaviortimes2frames import translate_times_to_frames

def consecutive_groups(iterable, ordering=lambda x: x):
    """Yield groups of consecutive items using :func:`itertools.groupby`.
    The *ordering* function determines whether two items are adjacent by
    returning their position.

    By default, the ordering function is the identity function. This is
    suitable for finding runs of numbers:

        >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
        >>> for group in consecutive_groups(iterable):
        ...     print(list(group))
        [1]
        [10, 11, 12]
        [20]
        [30, 31, 32, 33]
        [40]

    For finding runs of adjacent letters, try using the :meth:`index` method
    of a string of letters:

        >>> from string import ascii_lowercase
        >>> iterable = 'abcdfgilmnop'
        >>> ordering = ascii_lowercase.index
        >>> for group in consecutive_groups(iterable, ordering):
        ...     print(list(group))
        ['a', 'b', 'c', 'd']
        ['f', 'g']
        ['i']
        ['l', 'm', 'n', 'o', 'p']

    Each group of consecutive items is an iterator that shares it source with
    *iterable*. When an an output group is advanced, the previous group is
    no longer available unless its elements are copied (e.g., into a ``list``).

        >>> iterable = [1, 2, 11, 12, 21, 22]
        >>> saved_groups = []
        >>> for group in consecutive_groups(iterable):
        ...     saved_groups.append(list(group))  # Copy group elements
        >>> saved_groups
        [[1, 2], [11, 12], [21, 22]]

    """
    for k, g in itertools.groupby(
        enumerate(iterable), key=lambda x: x[0] - ordering(x[1])
    ):
        yield map(itemgetter(1), g)

def balance_classes(dlc_labels, behavior_matrix):
    outcomes       = [1 if outcome=='hit' else 0 for outcome in df['outcome'].tolist()]
    class_freq     = np.array(outcomes).mean()
    print(outcomes)
    print(class_freq)
    pass


def plot_corrected_labeled_frames(df):
    """
    Display all corrected data joint-by-joint.
    """
    xs_idxs = df.iloc[1].map(lambda x: str(x) == 'x')
    xs      = df[xs_idxs.index[xs_idxs]][2:].values.tolist()
    ys_idxs = df.iloc[1].map(lambda x: str(x) == 'y')
    ys      = df[ys_idxs.index[ys_idxs]][2:].values.tolist()
    data    = {'x': list(), 'y': list(), 'label': list()}
    fig     = plt.figure()

    for x_joints, y_joints in zip(xs, ys):
        joint   = 1
        for x, y in zip(x_joints, y_joints):
            data['x'].append(pandas.to_numeric(x))
            data['y'].append(pandas.to_numeric(y))
            data['label'].append(f"joint{str(joint)}")
            joint += 1
            if joint > len(x_joints):
                joint = 1

    data   = pandas.DataFrame(data)
    fig    = plt.figure()
    joint1 = data.loc[data['label'] == 'joint1']
    joint2 = data.loc[data['label'] == 'joint2']
    joint3 = data.loc[data['label'] == 'joint3']
    joint4 = data.loc[data['label'] == 'joint4']
    joint5 = data.loc[data['label'] == 'joint5']
    joint6 = data.loc[data['label'] == 'joint6']
    joint7 = data.loc[data['label'] == 'joint7']
    joint8 = data.loc[data['label'] == 'joint8']
    joints = [joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8]
    cmp    = iter(plt.cm.cool(np.linspace(0, 1, len(joints))))

    for joint in joints:
        label = joint['label'].iloc[0]
        plt.scatter(joint['x'], joint['y'], color=next(cmp), alpha=1, edgecolor='k', label=label)

    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.legend()
    fig.savefig("FullAdjustedData.svg")
    return


def nconsecutive(arr):
    """
    Finds non-consecutive values in iterable object.
    """
    for group in consecutive_groups(arr):
        group = list(group)

        if len(group) == 1:
            yield group[0]

        else:
            yield group[0], group[-1]


def get_equidistant_points(p1, p2, parts):
    """
    Finds n equidistant points between two points.
    """
    return list(zip(*[np.linspace(p1[i], p2[i], parts+2) for i in range(len(p1))]))


def interp(start, stop, num_bad_frames, df, headers):
    """
    Finds equidistant points between last and next good x,y values for all joints in df for all outliers.
    Returns corrected df for trial frames.
    """
    frames = [str(frame) for frame in range(start, stop + 1)]
    df     = df[df[0].isin(frames)]

    if df.shape[0] != len(frames):
        for i, row in df.iterrows():
            df.loc[i, 1:] = np.nan
        return df
    
    if len(frames) > 3:
        for i, row in df.iterrows():
            df.loc[i, 1:] = np.nan
        return df    

    df                     = pandas.concat([headers, df], ignore_index=True)
    cols                   = df.iloc[1].map(lambda x: str(x) != 'likelihood')
    cols                   = df[cols.index[cols]][2:]
    cols                   = cols.columns.values[:-3]
    xs_idxs                = df.iloc[1].map(lambda x: str(x) == 'x')
    xs                     = df[xs_idxs.index[xs_idxs]][2:].values.tolist()
    ys_idxs                = df.iloc[1].map(lambda x: str(x) == 'y')
    ys                     = df[ys_idxs.index[ys_idxs]][2:].values.tolist()
    last_good_row_x        = xs[0]
    next_good_row_x        = xs[-1]
    last_good_row_y        = ys[0]
    next_good_row_y        = ys[-1]
    adjusted_outliers      = defaultdict(list)

    for x1, y1, x2, y2 in zip(last_good_row_x, last_good_row_y, next_good_row_x, next_good_row_y):
        interpolated_outliers = get_equidistant_points((float(x1), float(y1)), (float(x2), float(y2)), num_bad_frames)
        
        for frame, coords in zip(frames, interpolated_outliers):
            for coord in coords:
                adjusted_outliers[frame].append(coord)

    adjusted_df         = pandas.DataFrame.from_dict(adjusted_outliers, orient='index').reset_index()
    adjusted_df.columns = cols

    return adjusted_df


def interpolate_outliers(df, outlier_df, headers):
    """
    Removes outlier data and replaces with interpolated x,y values for all joints.
    Returns further-pruned df.
    Removes all data from all trials with > 3 consecutive outlier frames or with outlier frames in first or last frame of trial (impossible to interpolate).
    """
    start_idx            = list(filter(None, [index if identity == 'start' else None for index, identity in enumerate(outlier_df.identity.tolist())]))
    stop_idx             = list(filter(None, [index if identity == 'stop' else None for index, identity in enumerate(outlier_df.identity.tolist())]))
    start_stop           = [(pandas.to_numeric(outlier_df[0][start]), pandas.to_numeric(outlier_df[0][stop])) for start, stop in zip(start_idx, stop_idx)]
    iso_idx              = list(filter(None, [index if identity == 'iso' else None for index, identity in enumerate(outlier_df.identity.tolist())]))
    isos                 = [start_stop.append((pandas.to_numeric(outlier_df[0][iso]), pandas.to_numeric(outlier_df[0][iso]))) for iso in iso_idx]
    last_next_good_frame = [(start - 1, stop + 1, stop - start + 1) for (start, stop) in start_stop]
    interpolated_dfs     = [interp(start, stop, num_bad_frames, df, headers) for (start, stop, num_bad_frames) in last_next_good_frame]
    interpolated_df      = pandas.concat([interp_df for interp_df in interpolated_dfs], ignore_index=True)
    df_idxs              = df.iloc[1].map(lambda x: str(x) != 'likelihood')
    df                   = df[df_idxs.index[df_idxs]][2:]

    for idx, row in interpolated_df.iterrows():
        frame         = row[0]
        index         = df.index[df[0] == str(frame)].tolist()
        index         = index[0]
        df.loc[index] = row

    df                 = df.drop('next_same', 1)
    df                 = df.drop('outliers', 1)
    df                 = df.drop('previous_same', 1)
    trial_ranges       = list(nconsecutive(pandas.to_numeric(df[0].values)))
    frames_to_drop     = []

    for idx, row in df.iterrows():
        frame  = int(row[0])
        is_nan = row[1:].isnull()
        is_nan = is_nan.all()

        if is_nan == True:
            for (start, stop) in trial_ranges:
                if start <= frame <= stop:
                    frames_to_drop.append([f for f in range(start, stop + 1)])

    frames_to_drop = sorted(list(set([f for s in frames_to_drop for f in s])))
    df['drop']     = [True if int(df.loc[i,0]) in frames_to_drop else False for i, r in df.iterrows()]
    df             = df.loc[df['drop'] == False]
    df             = pandas.concat([headers, df], ignore_index=True)
    df_idxs        = df.iloc[1].map(lambda x: str(x) != 'likelihood')
    df             = df[df_idxs.index[df_idxs]]
    return df


def replace_outliers(df, outliers):
    """
    Replaces outlier rows by averaging prior and following frame coordinates.
    If first or last frame, simply removes frames.
    """
    headers                = df.loc[:2,:]
    df                     = df.loc[2:,:]
    df['outliers']         = outliers
    df['previous_same']    = df.outliers.eq(df.outliers.shift())
    df['next_same']        = df.outliers.eq(df.outliers.shift(-1))
    outlier_df             = df[df.outliers == -1]
    outlier_df['identity'] = ['start' if (p == False and n == True) else \
                              'cont' if (p == n == True) else \
                              'stop' if (p == True and n == False) else \
                              'iso' for p, n in zip(outlier_df.previous_same.values, outlier_df.next_same.values)]
    outlier_df             = pandas.concat([headers, outlier_df], ignore_index=True)
    df                     = pandas.concat([headers, df], ignore_index=True)
    outlier_df_idxs        = outlier_df.iloc[1].map(lambda x: str(x) != 'likelihood')
    outlier_df             = outlier_df[outlier_df_idxs.index[outlier_df_idxs]]
    interpolated_df        = interpolate_outliers(df, outlier_df, headers)
    return interpolated_df


def prune_labels(dlc_labels, behavior_matrix, frame_ratio):
    """
    Removes DLC-traced frames outside of relevant window (from stimulus presentation to choice lick).
    """
    csv_out, ext           = os.path.splitext(dlc_labels)
    csv_out                = f"{csv_out}_pruned{ext}"
    df                     = translate_times_to_frames(behavior_matrix, frame_ratio)
    frames_to_keep         = [[s for s in range((start - 1), stop)] for start, stop in zip(df['start_frame'], df['rwin_frame'])]
    frames_to_keep         = [str(f) for startstop in frames_to_keep for f in startstop]
    labeled_frames         = pandas.read_csv(dlc_labels)
    headers                = pandas.DataFrame([labeled_frames.iloc[i,:].tolist() for i in range(2)])
    labeled_frames         = labeled_frames[labeled_frames['scorer'].isin(frames_to_keep)]
    labeled_frames.columns = headers.columns
    labeled_frames         = headers.append(labeled_frames)
    labeled_frames.to_csv(csv_out, index=False, header=None)
    return labeled_frames


def preprocess_behavior_and_labels(dlc_labels, behavior_matrix, frame_ratio=(200/300)):
    """
    Prunes DLC data to relevant frames.
    Calculates azimuth angle.
    Balances correct/incorrect classifiers (assumes S/R classifier balance).
    Saves figures to PWD.
    """
    labeled_frames      = prune_labels(dlc_labels, behavior_matrix, frame_ratio)
    principal_df        = detect_outliers.pca(labeled_frames)
    pruned_df, outliers = detect_outliers.isoForest(principal_df)
    #_                   = detect_outliers.assess_isoForest(labeled_frames, outliers)
    #_                   = detect_outliers.kmeans(labeled_frames)
    #_                   = detect_outliers.polar_kmeans(labeled_frames)
    #_                   = detect_outliers.SLINK(labeled_frames)
    _                   = detect_outliers.display_pruned_labels(labeled_frames, outliers)
    interpolated_df     = replace_outliers(labeled_frames, outliers)
    _                   = plot_corrected_labeled_frames(interpolated_df)
    _                   = plm(labeled_frames, outliers)
    return interpolated_df