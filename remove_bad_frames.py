import os
import pandas

def update_csv(csvpath, deleted_img):
    """
    Removes bad labels from DLC labels csv.
    """
    substring = 'labeled-data'
    path      = substring + deleted_img.split(substring)[1]
    df        = pandas.read_csv(csvpath, header=None)
    df        = df[~df[0].str.contains(path)]
    csv       = df.to_csv(csvpath, index=False, header=False)


def delete_labels(csvpath, img2labelpath, labeled_img):
    """
    Deletes images for labeling where labels are innacurate (as marked by user).
    csvpath:       Full path to DLC labels csv.
    img2labelpath: Full path to directory of images to be labeled.
    labeled_img:   Name *only* of labeled image (no path).
    """
    ext    = '.png'
    img    = labeled_img.split('_')[0] + ext
    file   = os.path.join(img2labelpath, img)
    remove = os.remove(file)
    update = update_csv(csvpath, file)