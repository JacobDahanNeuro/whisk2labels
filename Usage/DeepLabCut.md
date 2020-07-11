```bash
cd 'full/path/to/DLC/src/'
conda activate DLC
ipython
```

```python
# Prior to running whisk2labels. 
# Note: Edit number of frames to extract in config.yaml.
import deeplabcut
config_path = deeplabcut.create_new_project('project_name', 'scorer', ['path/to/video1.mkv','path/to/video2.mkv'...],working_directory='path/to/save/directory',copy_videos=True, videotype='.mkv')
deeplabcut.extract_frames(conﬁg_path)

# After running whisk2labels
# Note: Edit USER to reflect scorer name.
deeplabcut.convertcsv2h5(config_path, scorer='USER')
deeplabcut.check_labels(conﬁg_path)

# After running remove_bad_frames
deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(conﬁg_path, shuffle=1, displayiters=1000, saveiters=10000)
deeplabcut.evaluate_network(conﬁg_path, plotting=True)
deeplabcut.analyze_videos(config_path, ['path/to/video1.mkv','path/to/video2.mkv'...])
deeplabcut.plot_trajectories(config_path, ['path/to/video1.mkv','path/to/video2.mkv'...])
deeplabcut.create_labeled_video(config_path, ['path/to/video1.mkv','path/to/video2.mkv'...])

# Refinement
deeplabcut.extract_outlier_frames(config_path, ['path/to/video1.mkv','path/to/video2.mkv'...])
deeplabcut.reﬁne_labels(conﬁg_path)
deeplabcut.merge_datasets(conﬁg_path)
```
