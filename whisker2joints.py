import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def convert_whisker_to_joint_labels(x_coords, y_coords, n_joints):
    """
    Converts x- and y-coordinates into n equidistant joints.
    Returns df of n rows with columns y and x containing joint coordinates.
    """
    distances             = np.cumsum(np.sqrt( np.ediff1d(x_coords, to_begin=0)**2 + np.ediff1d(y_coords, to_begin=0)**2 ))
    point2point_distances = distances/distances[-1]
    fx, fy                = interp1d( point2point_distances, x_coords ), interp1d( point2point_distances, y_coords )
    alpha                 = np.linspace(0, 1, n_joints)
    x_out, y_out          = fx(alpha), fy(alpha)
    coords                = zip(x_out, y_out)
    df                    = pandas.DataFrame(coords, columns=('y', 'x'))
    return df
