```python
"""
Prior to usage:
    - Run WhiskiWrap: Trace video.
    - Run DLC:        Kmeans extract frames for labeling.
"""

# Change working directory
cd 'full/path/to/whisk/directory/'

h5            = 'full/path/to/h5.hdf5'
imagepath     = 'labeled-data/video_name/'
scorer        = 'UNI'
n_joints      = 8
csvpath       = 'full/path/to/csv.csv'
img2labelpath = 'full/path/to/labeled-data/video-name/'

# Identify C2 and segment into n_joints
from label_frames import *
find_and_segment_whisker(h5, imagepath, scorer, n_joints)

# Compare labeled frames against kmeans clustered frames for labeling from DLC; save only matches
kmeans(csvpath, imagepath, img2labelpath, scorer)

# Identify and remove poor whiski results w/ GUI
import remove_bad_frames_gui

remove_bad_frames_gui()

"""
To manually identify and remove poor results (no GUI):

from remove_bad_frames import *

# Manually identify poor whiski results
labeled_imgs = ['img000001_bodyparts.png', 'img000048_bodyparts.png', 'img001820_bodyparts.png', ...]

# Remove bad frames from DLC training set to preserve network integrity
for labeled_img in labeled_imgs:
  delete_labels(csvpath, img2labelpath, labeled_img)
"""

"""
Following usage:
    - Run DLC: convert csv to h5
"""
```  
