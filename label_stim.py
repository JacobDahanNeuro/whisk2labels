import os
import cv2
import sys
import pandas
import numpy as np
import tables as tb
from tables import *
from math import hypot
from builtins import int
from builtins import str
from builtins import list
from pandas.core.common import flatten
from stim2joints import convert_stim_to_joint_labels

class Stim(tb.IsDescription):
    x_coords = tb.Float128Col(shape=(2, 1))
    y_coords = tb.Float128Col(shape=(2, 1))


class Joints(tb.IsDescription):
    """
    Edit shape assignment as appropriate for n_joints.
    """
    x_coords = tb.Float128Col(shape=(20, 1))
    y_coords = tb.Float128Col(shape=(20, 1))


def fill_list(src_list, targ_len):
    """
    Takes a varible length list and returns a new list with a fixed length.
    """
    for i in range(targ_len):
        try:
            yield src_list[i]
        except IndexError:
            yield 0


def print_progress_bar(iteration, max_iters, post_text, bar_size=40):
    """
    Prints progress bar for given function.
    """
    j        = iteration/float(max_iters)
    percent  = str(100 * j)
    sys.stdout.write("\r" + " " * 9 + "[" + "=" * int(bar_size * j) + " " * (bar_size - int(bar_size * j)) + "]" + " " * 5 + percent + "%" + " " * 5 + post_text + "\r")
    sys.stdout.flush()


def display_and_label_stim(h5, f):
    """
    Display image for labling stim and save labels to hdf5.
    """
    refPt  = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x,y])
            cv2.drawMarker(img,(x,y),color=(0,0,255),markerType=cv2.MARKER_CROSS,thickness=2)
            cv2.imshow("Trace stim edges. Press any key to exit.", img)

    img = cv2.imread(f)
    
    cv2.imshow("Trace stim edges. Press any key to exit.", img)
    cv2.setMouseCallback("Trace stim edges. Press any key to exit.", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow("Trace stim edges. Press any key to exit.")
    cv2.destroyAllWindows()
    
    for i in range(5):    # maybe 5 or more
        cv2.waitKey(1)

    return refPt


def find_stim(h5, img2labelpath):
    """
    User-guided tracking of stimulus corners for all kmeans selected images.
    """
    directory          = img2labelpath
    pngs               = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and '.png' in f]
    full_path_pngs     = [os.path.join(directory, f) for f in pngs]
    coords             = [display_and_label_stim(h5, f) for f in full_path_pngs]
    h5file             = open_file(h5, mode="r+")
    table              = h5file.create_table(h5file.root, 'stim_corners', Stim, "Stim corners to track")

    for frame in coords:
        row      = table.row
        x_coords = []
        y_coords = []

        for xy in frame:
            x_coord, y_coord = xy
            x_coords.append(x_coord)
            y_coords.append(y_coord)

        row['x_coords'] = np.array(list(fill_list(x_coords, 2))).reshape((2,1))
        row['y_coords'] = np.array(list(fill_list(y_coords, 2))).reshape((2,1))
        row.append() 

    table.flush()
    h5file.close()


def convert_stim_to_joints(h5, n_joints):
    """
    Converts stim in each frame to equidistant joints.
    Saves joints to hdf5.
    """
    h5file         = open_file(h5, mode="r+")
    x_coords       = [list(flatten(row['x_coords'][:])) for row in h5file.root.stim_corners.iterrows()]
    y_coords       = [list(flatten(row['y_coords'][:])) for row in h5file.root.stim_corners.iterrows()]
    segmented_stim = []
    iteration      = 0

    for x, y in zip(x_coords, y_coords):
        max_iters  = len(x_coords)
        post_text  = 'Converting stim to joints.'
        all_coords = [x, y]
        all_coords = [item for sublist in all_coords for item in sublist]
        
        if 0.0 in all_coords:
            x_out  = [0.0 for _ in range(20)]
            y_out  = [0.0 for _ in range(20)]
            coords = zip(x_out, y_out)
            df     = pandas.DataFrame(coords, columns=('x', 'y'))
        
        else:
            df     = convert_stim_to_joint_labels(x, y, n_joints)
            
        x_labels   = df['x'].tolist()
        y_labels   = df['y'].tolist()
        stim       = [x_labels, y_labels]
        iteration += 1
        segmented_stim.append(stim)
        print_progress_bar(iteration, max_iters, post_text)


    print('\n         Finished converting stim to joints.')
    
    table     = h5file.create_table(h5file.root, 'stim_joints', Joints, "Segmented stim")
    row       = table.row
    iteration = 0

    for frame in segmented_stim:
        x_coords, y_coords = frame
        max_iters          = len(segmented_stim)
        post_text          = 'Appending stim joints to hdf5 file.'
        row['x_coords']    = np.array(x_coords).reshape((n_joints,1))
        row['y_coords']    = np.array(y_coords).reshape((n_joints,1))
        iteration         += 1
        row.append()
        print_progress_bar(iteration, max_iters, post_text)


    print('\n         Finished appending stim joints.')
    table.flush()
    h5file.close()


def joints_to_csv(h5, img2labelpath, scorer, n_joints):
    """
    Reads hdf5 table to DLC csv formatting. Combines with existing DLC csv for kmeans frames.
    Convert to hdf5 with DLC built-in deeplabcut.convertcsv2h5('path_to_config.yaml', scorer='experimenter').
    Saves to PWD.
    """
    h5file            = open_file(h5, mode="r+")
    csvpath           = os.path.join(img2labelpath, 'CollectedData_%s.csv' % scorer)
    csvfile           = pandas.read_csv(csvpath, header=None)
    x_coords          = [list(flatten(row['x_coords'][:])) for row in h5file.root.stim_joints.iterrows()]
    y_coords          = [list(flatten(row['y_coords'][:])) for row in h5file.root.stim_joints.iterrows()]
    frames            = [row for row in csvfile[0][3:]]
    labels_x          = ["stim_loc%s_x" % str(joint + 1) for joint in range(n_joints)]
    labels_y          = ["stim_loc%s_y" % str(joint + 1) for joint in range(n_joints)]
    labels            = [i for j in zip(labels_x, labels_y) for i in j]
    labels            = ['bodyparts'] + labels[:]
    dfx               = pandas.DataFrame(x_coords, columns=labels_x)
    dfy               = pandas.DataFrame(y_coords, columns=labels_y)
    dfconcat          = pandas.concat([dfx,dfy]).sort_index().reset_index(drop=True)
    data              = {c: dfconcat[c].dropna().values for c in dfconcat.columns}
    data['bodyparts'] = frames
    df                = pandas.DataFrame(data, columns=labels)
    labels_x          = ["stim_loc%s" % str(joint + 1) for joint in range(n_joints)]
    labels_y          = ["stim_loc%s" % str(joint + 1) for joint in range(n_joints)]
    labels            = [i for j in zip(labels_x, labels_y) for i in j]
    labels            = ['bodyparts'] + labels[:]
    x_tags            = ["x" for i in range(n_joints)]
    y_tags            = ["y" for i in range(n_joints)]
    xy_tags           = [i for j in zip(x_tags, y_tags) for i in j]
    xy_tags           = ['coords'] + xy_tags[:]
    df                = pandas.DataFrame(np.insert(df.values, 0, values=xy_tags, axis=0))
    df                = pandas.DataFrame(np.insert(df.values, 0, values=labels, axis=0))
    headers           = ['scorer'] + [scorer for label in range(len(labels)-1)]
    df.columns        = [labels]
    df                = pandas.DataFrame(np.insert(df.values, 0, values=headers, axis=0))
    df                = df.drop(columns=0).reindex()
    df                = pandas.concat([csvfile, df], axis=1)
    df[df.eq(0)]      = np.nan
    csv               = df.to_csv('joints_for_dlc.csv', index=False, header=False)
    h5file.close()


def find_and_segment_stim(h5, img2labelpath, scorer, n_joints=20):
    """
    User-guided tracking of stimulus edge.    
    h5:            Full path to h5 file.
    img2labelpath: Full path to directory of images to be labeled (e.g., .../labeled-data/video-name).
    scorer:        Name of scorer to be used in DLC labeling.
    n_joints:      Number of joints for whisker segmentation.
    """
    find_stim(h5, img2labelpath)
    convert_stim_to_joints(h5, n_joints)
    joints_to_csv(h5, img2labelpath, scorer, n_joints)
