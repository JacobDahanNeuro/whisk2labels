import os
import sys
import pandas
import numpy as np
from tables import *
from math import hypot
from builtins import int
from builtins import str
from builtins import list
from pandas.core.common import flatten
from whisker2joints import convert_whisker_to_joint_labels


class Whisker(IsDescription):
    x_coords = Float128Col(shape=(1000, 1))
    y_coords = Float128Col(shape=(1000, 1))
    time     = Int32Col()


class Joints(IsDescription):
    """
    Edit shape assignment as appropriate for n_joints.
    """
    x_coords = Float128Col(shape=(8, 1))
    y_coords = Float128Col(shape=(8, 1))
    time     = Int32Col()


def print_progress_bar(iteration, max_iters, post_text, bar_size=40):
    """
    Prints progress bar for given function.
    """
    j        = iteration/float(max_iters)
    percent  = str(100 * j)
    sys.stdout.write("\r" + " " * 8 + "[" + "=" * int(bar_size * j) + " " * (bar_size - int(bar_size * j)) + "]" + " " * 5 + percent + "%" + " " * 5 + post_text + "\r")
    sys.stdout.flush()


def fill_list(src_list, targ_len):
    """
    Takes a varible length list and returns a new list with a fixed length.
    """
    for i in range(targ_len):
        try:
            yield src_list[i]
        except IndexError:
            yield 0


def whisker_length(whisker):
    """
    Determines euclidian whisker length from list of coordinates.
    Returns euclidian length.
    """
    x_coords = whisker[1]
    y_coords = whisker[2]
    coords   = zip(x_coords, y_coords)
    ptdiff   = lambda (p1,p2): (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    diffs    = (ptdiff((p1, p2)) for p1, p2 in zip (coords, coords[1:]))
    path     = sum(hypot(*d) for d in  diffs)
    return path


def find_longest(whiskers):
    """
    Finds longest whisker in list of whisker coordinates.
    Returns list of longest whisker coordinates.
    """
    whisker_lengths = [whisker_length(whisker) for whisker in whiskers]
    longest_whisker = whiskers[whisker_lengths.index(max(whisker_lengths))]
    return longest_whisker


def find_whisker(h5):
    """
    Predict which traced whisker is C2 by finding longest whisker in each frame.
    Returns new hdf5 table with row tracing all C2 x and y coords for each frame.
    """
    h5file             = open_file(h5, mode="r+")
    x_coords           = [row.tolist() for row in h5file.root.pixels_x.iterrows()]
    y_coords           = [row.tolist() for row in h5file.root.pixels_y.iterrows()]
    times              = [int(row['time']) for row in h5file.root.summary.iterrows()]
    unique_times, idxs = np.unique((np.array(times)), return_index=True)
    unique_times, idxs = list(unique_times), list(idxs)
    whiskers_to_trace  = []

    for idx, unique_time in enumerate(unique_times):
        iteration        = unique_times.index(unique_time)
        max_iters        = len(unique_times)
        post_text        = 'Finding longest whiskers.'
        start            = idxs[idx]
        try:
            stop         = idxs[idx + 1] - 1
        except IndexError:
            stop         = times.index(times[-1])
        whiskers         = [[time, x, y] for x, y, time in zip(x_coords[start:stop+1], y_coords[start:stop+1], times[start:stop+1])]
        longest_whisker  = find_longest(whiskers)
        whiskers_to_trace.append(longest_whisker)
        print_progress_bar(iteration, max_iters, post_text)
    
    print('\nFinished finding longest whiskers.')

    table   = h5file.create_table(h5file.root, 'longest', Whisker, "Whiskers to track")
    whisker = table.row

    for frame in whiskers_to_trace:
        iteration           = frame.index(whiskers_to_trace)
        max_iters           = len(whiskers_to_trace)
        post_text           = 'Appending longest whiskers to hdf5 file.'
        whisker['time']     = frame[0]
        whisker['x_coords'] = np.array(list(fill_list(frame[1], 1000))).reshape((1000,1))
        whisker['y_coords'] = np.array(list(fill_list(frame[2], 1000))).reshape((1000,1))
        whisker.append()
        print_progress_bar(iteration, max_iters, post_text)
            
    table.flush()
    h5file.close()


def convert_to_joints(h5, n_joints):
    """
    Converts whisker in each frame to equidistant joints.
    Saves joints to hdf5.
    """
    h5file             = open_file(h5, mode="r+")
    x_coords           = [filter(lambda a: a != 0.0, list(flatten(row['x_coords'][:]))) for row in h5file.root.longest.iterrows()]
    y_coords           = [filter(lambda a: a != 0.0, list(flatten(row['y_coords'][:]))) for row in h5file.root.longest.iterrows()]
    times              = [int(row['time']) for row in h5file.root.longest.iterrows()]
    segmented_whiskers = []

    for x, y, time in zip(x_coords, y_coords, times):
        iteration = times.index(time)
        max_iters = len(times)
        post_text = 'Converting whiskers to joints.'
        df        = convert_whisker_to_joint_labels(x, y, n_joints)
        x_labels  = df['x'].tolist()
        y_labels  = df['y'].tolist()
        whisker   = [time, x_labels, y_labels]
        segmented_whiskers.append(whisker)
        print_progress_bar(iteration, max_iters, post_text)
    
    print('\nFinished converting whiskers to joints.')
    
    table   = h5file.create_table(h5file.root, 'joints', Joints, "Segmented whiskers")
    whisker = table.row

    for frame in segmented_whiskers:
        iteration           = frame.index(segmented_whiskers)
        max_iters           = len(segmented_whiskers)
        post_text           = 'Appending whisker joints to hdf5 file.'
        whisker['time']     = frame[0]
        whisker['x_coords'] = np.array([coord for coord in frame[1]]).reshape((n_joints,1))
        whisker['y_coords'] = np.array([coord for coord in frame[2]]).reshape((n_joints,1))
        whisker.append()
        print_progress_bar(iteration, max_iters, post_text)
        
    table.flush()
    h5file.close()


def joints_to_csv(h5, imagepath, scorer, n_joints):
    """
    Reads hdf5 table to DLC csv formatting. Convert to hdf5 with DLC built-in deeplabcut.convertcsv2h5('path_to_config.yaml', scorer='experimenter').
    Saves to PWD.
    """
    h5file            = open_file(h5, mode="r+")
    x_coords          = [list(flatten(row['x_coords'][:])) for row in h5file.root.joints.iterrows()]
    y_coords          = [list(flatten(row['y_coords'][:])) for row in h5file.root.joints.iterrows()]
    frames            = [str(imagepath + "img%06.f.png" % int(row['time'])) for row in h5file.root.joints.iterrows()]
    labels_x          = ["joint%s_x" % str(joint + 1) for joint in xrange(n_joints)]
    labels_y          = ["joint%s_y" % str(joint + 1) for joint in xrange(n_joints)]
    labels            = [i for j in zip(labels_x, labels_y) for i in j]
    labels            = ['bodyparts'] + labels[:]
    dfx               = pandas.DataFrame(x_coords, columns=labels_x)
    dfy               = pandas.DataFrame(y_coords, columns=labels_y)
    dfconcat          = pandas.concat([dfx,dfy]).sort_index().reset_index(drop=True)
    data              = {c: dfconcat[c].dropna().values for c in dfconcat.columns}
    data['bodyparts'] = frames
    df                = pandas.DataFrame(data, columns=labels)
    labels_x          = ["joint%s" % str(joint + 1) for joint in xrange(n_joints)]
    labels_y          = ["joint%s" % str(joint + 1) for joint in xrange(n_joints)]
    labels            = [i for j in zip(labels_x, labels_y) for i in j]
    labels            = ['bodyparts'] + labels[:]
    x_tags            = ["x" for i in xrange(n_joints)]
    y_tags            = ["y" for i in xrange(n_joints)]
    xy_tags           = [i for j in zip(x_tags, y_tags) for i in j]
    xy_tags           = ['coords'] + xy_tags[:]
    df                = pandas.DataFrame(np.insert(df.values, 0, values=xy_tags, axis=0))
    df                = pandas.DataFrame(np.insert(df.values, 0, values=labels, axis=0))
    headers           = ['scorer'] + ['jbd2144' for label in xrange(len(labels)-1)]
    df.columns        = [labels]
    df                = pandas.DataFrame(np.insert(df.values, 0, values=headers, axis=0))
    csv               = df.to_csv('joints_for_dlc.csv', index=False, header=False)
    h5file.close()


def kmeans(csvpath, img2labelpath, scorer):
    """
    Trims duplicate csv to contain only data for kmeans clustered frames designated for labeling.
    Saves csv to img2labelpath.
    csvpath:       Full path to csv labels file.
    img2labelpath: Full path to directory of images to be labeled (e.g., .../labeled-data/video-name).
    scorer:        Name of scorer designated for DLC labeling.
    """
    df             = pandas.read_csv(csvpath, header=None)
    kmeans         = [row[0] for index, row in df.iterrows() if os.path.split(row[0])[1] in os.listdir(img2labelpath)]
    rows           = [row[0] for index, row in df.iterrows()]
    missing        = [frame for frame in os.listdir(img2labelpath) if os.path.join(os.path.split(os.path.split(img2labelpath)[0])[1], os.path.split(img2labelpath)[1], frame) not in rows]
    remove         = [os.remove(os.path.join(img2labelpath, no_labels)) for no_labels in missing]
    df_save        = df[df[0].isin(kmeans)]
    df_save.loc[0] = df.loc[0]
    df_save.loc[1] = df.loc[1]
    df_save.loc[2] = df.loc[2]
    df_save        = df_save.sort_index()
    csv            = df_save.to_csv(os.path.join(img2labelpath, 'CollectedData_%s.csv' % scorer), index=False, header=False)


def find_and_segment_whisker(h5, imagepath, scorer, n_joints=8):
    """
    Finds C2 whisker and segments into n joints.
    h5:        Full path to h5 file.
    imagepath: Relative (expected) path to labeling directory (e.g. labeled-data/video_name/)
    scorer:    Name of scorer to be used in DLC labeling.
    n_joints:  Number of joints for whisker segmentation.
    """
    find_whisker(h5)
    convert_to_joints(h5, n_joints)
    joints_to_csv(h5, imagepath, scorer, n_joints)
