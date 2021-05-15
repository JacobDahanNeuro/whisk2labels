import pandas
import numpy as np
import matplotlib.pyplot as plt

def plm(df, outliers):


    headers = df.loc[:2,:]
    df      = df.loc[2:,:]
    df      = df.loc[[True if outlier==1 else False for outlier in outliers]]
    df      = pandas.concat([headers, df], ignore_index=True)
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
    cmp    = iter(plt.cm.plasma(np.linspace(0, 1, len(joints))))

    im     = plt.imread('/Users/jakedahan/Documents/Columbia/Bruno/Co1939/whisker+stim/labeled-data/JP61_20190411_reoriented/img013477.png')
    implot = plt.imshow(im)

    for joint in joints:
        label = joint['label'].iloc[0]
        plt.scatter(joint['x'], joint['y'], color=next(cmp), alpha=1, edgecolor='k', label=label)
    
    plt.axis('off')
    fig.savefig("MouseOverlay.svg", bbox_inches='tight')

    return