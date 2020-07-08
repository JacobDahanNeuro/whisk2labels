```bash
cd 'full/path/to/whisk/src'
conda activate whiski_wrap
ipython
```

```python
import ffmpeg
import os
import WhiskiWrap

input_vid = WhiskiWrap.FFmpegReader('path/to/video.mkv')
tiff_dir  = 'path/to/tiff/save/dir'
h5_fi     = 'path/to/save/dir/h5.hdf5'
n         = 10
side      = 'left/right'

WhiskiWrap.interleaved_reading_and_tracing(input_vid,tiff_dir, h5_filename=h5_fi, n_trace_processes=n, face=side)
results_summary = WhiskiWrap.read_whiskers_hdf5_summary(h5_fi)
measurements    = [WhiskiWrap.measure_chunk(os.path.join(os.getcwd(), '{}'.format(fi)), face=side) for fi in os.listdir('.') if "whiskers" in fi]
```