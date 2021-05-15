import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import fcluster, linkage


def cart2pol(x, y):
    """
    Converts cartesian coordinates to polar coordinates.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def SLINK(df):
    """
    Single linkage clustering on original x,y coordinate plane.
    """
    xs_idxs      = df.iloc[1].map(lambda x: str(x) == 'x')
    xs           = df[xs_idxs.index[xs_idxs]][2:].values.tolist()
    xs           = [float(x) for x_list in xs for x in x_list]
    ys_idxs      = df.iloc[1].map(lambda x: str(x) == 'y')
    ys           = df[ys_idxs.index[ys_idxs]][2:].values.tolist()
    ys           = [float(y) for y_list in ys for y in y_list]
    df           = pandas.DataFrame({'x': xs, 'y': ys})
    X            = df.to_numpy()
    Z            = linkage(X, 'ward')
    k            = 3
    clusters     = fcluster(Z, k, criterion='maxclust')
    colors       = ['fuchsia', 'gold', 'cyan']
    label_colors = [colors[c - 1] for c in clusters]
    fig          = plt.figure()
    
    plt.scatter(X[:,0], X[:,1], c=label_colors, edgecolor='k')
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    fig.savefig('SLINK.svg')
    return


def kmeans(df):
    """
    K-Means clustering of whisker traces.
    Note: Data violates K-Means assumptions.
        (1) Clusters are not evenly sized.
        (2) Clusters are not spherical.
    """
    xs_idxs      = df.iloc[1].map(lambda x: str(x) == 'x')
    xs           = df[xs_idxs.index[xs_idxs]][2:].values.tolist()
    xs           = [float(x) for x_list in xs for x in x_list]
    ys_idxs      = df.iloc[1].map(lambda x: str(x) == 'y')
    ys           = df[ys_idxs.index[ys_idxs]][2:].values.tolist()
    ys           = [float(y) for y_list in ys for y in y_list]
    df           = pandas.DataFrame({'x': xs, 'y': ys})
    kmeans       = KMeans(n_clusters=3)
    kmeans.fit(df)
    
    labels       = kmeans.predict(df)
    centroids    = kmeans.cluster_centers_
    clusters_    = ['Cluster1', 'Cluster2', 'Cluster3']
    colors       = ['fuchsia', 'cyan', 'gold']
    label_colors = [colors[label - 1] for label in labels]
    fig          = plt.figure()
    cluster1     = df.loc[[True if l == 0 else False for l in labels]]
    cluster2     = df.loc[[True if l == 1 else False for l in labels]]
    cluster3     = df.loc[[True if l == 2 else False for l in labels]]
    clusters     = [cluster1, cluster2, cluster3]

    for idx, cluster in enumerate(clusters):
        plt.scatter(cluster['x'], cluster['y'], c=colors[idx], edgecolor='k', label=clusters_[idx])
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.legend()
    fig.savefig('KMeans.svg')
    return


def polar_kmeans(df):
    """
    K-Means clustering of whisker traces on polar coordinate system.
    Attempt to resolve fundamental K-Means assumption violations.
    """
    xs_idxs   = df.iloc[1].map(lambda x: str(x) == 'x')
    xs        = df[xs_idxs.index[xs_idxs]][2:].values.tolist()
    xs        = [float(x) for x_list in xs for x in x_list]
    ys_idxs   = df.iloc[1].map(lambda x: str(x) == 'y')
    ys        = df[ys_idxs.index[ys_idxs]][2:].values.tolist()
    ys        = [float(y) for y_list in ys for y in y_list]
    df        = pandas.DataFrame({'x': xs, 'y': ys})
    rhos      = []
    phis      = []

    for x, y in zip(xs, ys):
        rho, phi = cart2pol(x, y)
        rhos.append(rho)
        phis.append(phi)
    
    df_polar     = pandas.DataFrame({'rho': rhos, 'phi': phis})
    kmeans       = KMeans(n_clusters=3)
    kmeans.fit(df_polar)
    labels       = kmeans.predict(df_polar)
    colors       = ['gold', 'cyan', 'fuchsia']
    label_colors = [colors[label - 1] for label in labels]
    centroids    = kmeans.cluster_centers_
    fig_polar    = plt.figure()

    plt.scatter(df_polar['rho'], df_polar['phi'], c=label_colors, edgecolor='k')
    plt.xlabel("rho")
    plt.ylabel("phi")
    fig_polar.savefig('KMeansPolar.svg')
    
    fig_cart = plt.figure()
    plt.scatter(df['x'], df['y'], c=label_colors, edgecolor='k')
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    fig_cart.savefig('KMeansPolarXY.svg')
    return


def pca(df, n_components=2):
    """
    Performs PCA to reduce whisker vectors to two dimensions.
    """
    xs_idxs                = df.iloc[1].map(lambda x: str(x) == 'x')
    xs                     = df[xs_idxs.index[xs_idxs]][2:]
    ys_idxs                = df.iloc[1].map(lambda x: str(x) == 'y')
    ys                     = df[ys_idxs.index[ys_idxs]][2:]
    features               = pandas.concat([xs, ys], axis=1)
    pca                    = PCA(n_components)
    principal_components   = pca.fit_transform(features)
    explained_variance     = pca.explained_variance_ratio_
    sum_explained_variance = np.sum(explained_variance) * 100
    principal_df           = pandas.DataFrame(data=principal_components, columns=['PC1','PC2'])
    fig                    = plt.figure()
    
    plt.scatter(principal_df['PC1'], principal_df['PC2'], c='cyan', alpha=1, edgecolor='k', label='Inlier')
    plt.scatter([],[], c='fuchsia', alpha=1, edgecolor='k', label='Outlier')
    plt.xlabel("Principal Component 1 (Arbitrary Units)")
    plt.ylabel("Principal Component 2 (Arbitrary Units)")
    plt.legend()
    fig.savefig('PCA.svg')
    return principal_df


def isoForest(df, contamination=0.035):
    """
    Identifies and plots outlier coordinates on PCA coordinate axis.
    """
    model         = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination, max_features=df.shape[1])
    model.fit(df)
    outliers      = model.predict(df)
    df['outlier'] = outliers
    pruned_df     = df[df['outlier'].map(lambda x: int(x) == 1)]
    outlier_df    = df[~df['outlier'].map(lambda x: int(x) == 1)]
    fig           = plt.figure()

    plt.scatter(pruned_df['PC1'], pruned_df['PC2'], c='cyan', alpha=1, edgecolor='k', label='Inlier')
    plt.scatter(outlier_df['PC1'], outlier_df['PC2'], c='fuchsia', alpha=1, edgecolor='k', label='Outlier')
    plt.xlabel("Principal Component 1 (Arbitrary Units)")
    plt.ylabel("Principal Component 2 (Arbitrary Units)")
    fig.savefig('IsolationForest.svg')
    return pruned_df, outliers


def assess_isoForest(df, outliers):
    """
    Display outliers on original (x,y) coordinate plane from DLC whisker trace.
    """
    colors  = ['cyan' if outlier==1 else 'fuchsia' for outlier in outliers]
    xs_idxs = df.iloc[1].map(lambda x: str(x) == 'x')
    xs      = df[xs_idxs.index[xs_idxs]][2:].values.tolist()
    ys_idxs = df.iloc[1].map(lambda x: str(x) == 'y')
    ys      = df[ys_idxs.index[ys_idxs]][2:].values.tolist()
    data    = {'x': list(), 'y': list(), 'color': list(), 'label': list()}
    fig     = plt.figure()

    for x_joints, y_joints, color in zip(xs, ys, colors):
        joint   = 1
        for x, y in zip(x_joints, y_joints):
            data['x'].append(pandas.to_numeric(x))
            data['y'].append(pandas.to_numeric(y))
            data['color'].append(color)
            data['label'].append(f"joint{str(joint)}")
            joint += 1
            if joint > len(x_joints):
                joint = 1

    plt.scatter(data['x'], data['y'], c=data['color'], alpha=1, edgecolor='k')
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    fig.savefig('IsolationForestXY.svg')


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
    fig.savefig('FullData.svg')
    return


def display_pruned_labels(df, outliers):
    """
    Display only inlier data joint-by-joint.
    """
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
    cmp    = iter(plt.cm.cool(np.linspace(0, 1, len(joints))))

    for joint in joints:
        label = joint['label'].iloc[0]
        plt.scatter(joint['x'], joint['y'], color=next(cmp), alpha=1, edgecolor='k', label=label)
    
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.legend()
    fig.savefig("NoOutliers.svg")

    return