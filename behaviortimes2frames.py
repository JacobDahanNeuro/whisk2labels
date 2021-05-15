import os
import math
import pandas
from collections import defaultdict 


def normal_round(n):
    """
    Performs "0.5 always up" rounding.
    """
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def time_to_frame(time, frame_ratio=(200/30)):
    """
    Converts time (s) to frame count given a real-time frame rate to video frame rate ratio.
    """
    frame_num = time * frame_ratio
    frame_num = normal_round(frame_num)
    return frame_num


def translate_times_to_frames(csvpath, frame_ratio=(200/30)):
    """
    Translates behavioral matrix times (s) to frames for pairing with DLC.
    """
    df           = pandas.read_csv(csvpath)
    df           = df[df['isrnd'].map(lambda x: str(x) == 'True')]
    df           = df[df['outcome'].map(lambda x: str(x) != 'curr')]
    df           = df.dropna()
    columns      = ['start_time', 'release_time', 'choice_time', 'rwin_time']
    times        = [df[c].tolist() for c in columns]
    new_columns  = ['start_frame', 'release_frame', 'choice_frame', 'rwin_frame']
    drop_columns = ['dirdel', 'servo_pos', 'stepper_pos']
    frames       = [[time_to_frame(t) for t in time] for time in times]
    
    for f, c in zip(frames, new_columns):
        df[c] = f

    for d in drop_columns:
        del df[d]

    return df