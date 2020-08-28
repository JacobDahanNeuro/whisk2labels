# whisk2labels
Extract C2 whisker from whiski data; convert whisker to joints.
For full pipeline, please view [full_pipe.md](https://github.com/JacobDahan/whisk2labels/blob/master/Usage/full_pipe.md).

## NB
1. Must use python2 conda environment (e.g., whiski_wrap).
2. PYTHONPATH must be exported in every new kernel.
3. remove_bad_frames is now deprecated in favor of remove_bad_frames_gui

## Add to usr path
```bash
cd ~
mkdir src
cd src
git clone https://github.com/JacobDahan/whisk2labels.git
# Note that USERNAME must be changed to individual username
export PYTHONPATH=$PYTHONPATH:/home/USERNAME/src/whisk2labels/
```

## find_and_segment_whisker
Finds C2 whisker and segments into n joints.
Saves output csv to present working directory.

Usage:
```python
from label_frames import *
find_and_segment_whisker(h5, imagepath, scorer, n_joints)
```

Inputs:
- h5:        Full path to h5 file.
- imagepath: Relative (expected) path to labeling directory (e.g. labeled-data/video_name/)
- scorer:    Name of scorer to be used in DLC labeling.
- n_joints:  Number of joints for whisker segmentation.

## kmeans
Trims duplicate csv to contain only data for kmeans clustered frames designated for labeling by DLC.

Usage:
```python
from label_frames import *
kmeans(csvpath, imagepath, img2labelpath, scorer)
```

Inputs:
- csvpath:       Full path to csv labels file.
- imagepath:     Relative path to labeling directory (e.g. labeled-data/video_name/)
- img2labelpath: Full path to directory of images to be labeled (e.g., .../labeled-data/video-name).
- scorer:        Name of scorer designated for DLC labeling.

## delete_labels
Deletes images for labeling where labels are innacurate (as marked by user).

Usage:
```python
# Manual version
from remove_bad_frames import *
delete_labels(csvpath, img2labelpath, labeled_img)

# Identify and remove poor whiski results w/ GUI (must use DLC conda environment)
from remove_bad_frames_gui import *
search_new_dir=True
find_bad_frames(search_new_dir)
```

Inputs (manual version):
- csvpath:        Full path to DLC labels csv.
- img2labelpath: Full path to directory of images to be labeled.
- labeled_img:    Name *only* of labeled image (no path).

## Full usage
```python
"""
Prior to usage:
    - Run WhiskiWrap: Trace video.
    - Run DLC:        Kmeans extract frames for labeling.
"""

# Change working directory
cd 'full/path/to/whisk/directory/'

h5            = 'full/path/to/h5.hdf5'
imagepath     = 'labeled-images/video_name/'
scorer        = 'UNI'
n_joints      = 8
csvpath       = 'full/path/to/csv.csv'
img2labelpath = 'full/path/to/labeled-data/video-name/'

# Identify C2 and segment into n_joints
from label_frames import *
find_and_segment_whisker(h5, imagepath, scorer, n_joints)

# Compare labeled frames against kmeans clustered frames for labeling from DLC; save only matches
kmeans(csvpath, imagepath, img2labelpath, scorer)

# Remove bad frames from DLC training set to preserve network integrity
# Identify and remove poor whiski results w/ GUI (must use DLC conda environment)
from remove_bad_frames_gui import *

search_new_dir=True

find_bad_frames(search_new_dir)
```
