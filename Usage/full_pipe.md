# Full sample pipeline for single-mouse network 

## Bash
```bash
# Add to user path
cd ~
mkdir src
cd src
git clone https://github.com/JacobDahan/whisk2labels.git
# Note that USERNAME must be changed to individual username
export PYTHONPATH=$PYTHONPATH:/home/USERNAME/src/whisk2labels/

# create dir structure
cd ~

mkdir my_current_project
cd my_current_project

mkdir whiski
cd whiski

mkdir my_current_mouse
cd my_current_mouse

# move videos into proper paths
mv /path/to/my_current_mouse_video.mkv path/to/my_current_project/whiski/my_current_mouse/my_current_mouse_video.mkv

# begin WhiskiWrap
conda activate whiski_wrap

ipython
```

## Python
```python
import WhiskiWrap
import ffmpeg
import os

"""
NB: It may be preferable to rotate all videos such that the mouse is facing 'upward.'
For near-lossless rotation (in bash): ffmpeg -i in.mkv -vf "transpose=2,transpose=2" -crf 12 -c:a copy -vcodec libx264 out.mkv
"""

input_vid = WhiskiWrap.FFmpegReader('path/to/my_current_project/whiski/my_current_mouse/my_current_mouse_video.mkv')
tiff_dir  = 'path/to/my_current_project/whiski/my_current_mouse'
h5_fi     = 'path/to/my_current_project/whiski/my_current_mouse/my_current_mouse_video.hdf5'
n         = 10
side      = 'left'
#side     = 'right'

WhiskiWrap.interleaved_reading_and_tracing(input_vid,tiff_dir, h5_filename=h5_fi, n_trace_processes=n)

results_summary = WhiskiWrap.read_whiskers_hdf5_summary(h5_fi)
measurements    = [WhiskiWrap.measure_chunk(os.path.join(tiff_dir, '{}'.format(fi)), face=side) for fi in os.listdir(tiff_dir) if "whiskers" in fi]
exit()
```

## Bash
```bash
conda deactivate

cd ~
cd my_current_project

# begin DLC
conda activate DLC

ipython
```

## Python
```python
import deeplabcut

project_name = 'my-current-DLC-project'
scorer       = 'my-UNI'
config_path  = deeplabcut.create_new_project(project_name, scorer, ['path/to/my_current_project/whiski/my_current_mouse/my_current_mouse_video.mkv'], working_directory='path/to/my_current_project', copy_videos=True, videotype='.mkv')

"""
NB: Update config.yaml numframes2pick before following step. 
Recommended numframes2pick: ~100 / n_mice 
"""

deeplabcut.extract_frames(conﬁg_path)

exit()
```

## Bash
```bash
conda deactivate

# begin whisk2labels
conda activate whiski_wrap

ipython
```

## Python
```python
cd 'path/to/my_current_project/whiski/my_current_mouse/'

h5            = 'path/to/my_current_project/whiski/my_current_mouse/my_current_mouse_video.hdf5'
imagepath     = 'labeled-data/my_current_mouse_video/'
scorer        = 'my-UNI'
n_joints      = 8
csvpath       = 'path/to/my_current_project/my-current-DLC-project/my_current_mouse/joints_for_dlc.csv'
img2labelpath = 'path/to/my_current_project/my-current-DLC-project/labeled-data/my_current_mouse/'

# Identify C2 and segment into n_joints
from label_frames import *

find_and_segment_whisker(h5, imagepath, scorer, n_joints)

# Compare labeled frames against kmeans clustered frames for labeling from DLC; save only matches
kmeans(csvpath, imagepath, img2labelpath, scorer)

exit()
```

## Bash
```bash
conda deactivate

# begin DLC
conda activate DLC

ipython
```

## Python
```python
import deeplabcut

"""
NB: Update config.yaml label names before the following steps. 
Correct labels (for n joints):
-joint1
-joint2
...
-jointn
"""

config_path = 'path/to/my_current_project/my-current-DLC-project/config.yaml'

# check whiski + whisk2labels labels
deeplabcut.convertcsv2h5(config_path, scorer='my-UNI')
deeplabcut.check_labels(conﬁg_path)

# create + train + evaluate network
deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(conﬁg_path, shuffle=1, displayiters=1000, saveiters=10000)
deeplabcut.evaluate_network(conﬁg_path, plotting=True)
deeplabcut.analyze_videos(config_path, ['path/to/my_current_project/my-current-DLC-project/videos/my_current_mouse/my_current_mouse_video.mkv'])
deeplabcut.plot_trajectories(config_path, ['path/to/my_current_project/my-current-DLC-project/videos/my_current_mouse/my_current_mouse_video.mkv'])
deeplabcut.create_labeled_video(config_path, ['path/to/my_current_project/my-current-DLC-project/videos/my_current_mouse/my_current_mouse_video.mkv'])

# refine network
deeplabcut.extract_outlier_frames(config_path, ['path/to/my_current_project/my-current-DLC-project/videos/my_current_mouse/my_current_mouse_video.mkv'])
deeplabcut.reﬁne_labels(conﬁg_path)
deeplabcut.merge_datasets(conﬁg_path)

exit()
```
