import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def convert_whisker_to_joint_labels(x_coords, y_coords, n_joints):

    distance = np.cumsum(np.sqrt( np.ediff1d(x_coords, to_begin=0)**2 + np.ediff1d(y_coords, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, x_coords ), interp1d( distance, y_coords )

    alpha                = np.linspace(0, 1, n_joints)
    x_regular, y_regular = fx(alpha), fy(alpha)

    coords = zip(y_regular, x_regular)
    df     = pandas.DataFrame(coords, columns=('y', 'x'))

    return df