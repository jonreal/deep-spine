"""Time-lapse of a whole run's sim.mp4 (Movie S1: the canonical run at 2000x).

Usage (repo root, venv active):
    python analysis/pubfigs/timelapse_movie.py <path/to/sim.mp4> [out.mp4] [speed] [fps] [crf]
Defaults: out = analysis/pubfigs/out/si-movie-full.mp4, speed = 2000 (x real
time), fps = 30, crf = 18 (x264 quality; higher = smaller file: 18 gives ~60 MB
for the canonical run). The canonical run (2000 x 90 s = 50 h) gives a 90 s,
2700-frame movie.

sim.mp4 is written at 10x real time (container rate 10*frame_rate), so the
playback timestamps are divided by speed/10 and the result is resampled to fps
(frames in between are dropped). Decodes the whole render: ~7 min for the
canonical sim.mp4 on the Mac Studio used for the paper. Needs ffmpeg.
"""
import os
import subprocess
import sys

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join('analysis', 'pubfigs', 'out', 'si-movie-full.mp4')
speed = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.0
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
crf = int(sys.argv[5]) if len(sys.argv) > 5 else 18     # 18 as cut_movies.py
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

subprocess.run(
    ['ffmpeg', '-y', '-v', 'error', '-stats', '-i', src,
     '-vf', f'setpts=PTS/{speed / 10:g}', '-r', f'{fps:g}',
     '-c:v', 'libx264', '-crf', str(crf), '-preset', 'slow',
     '-pix_fmt', 'yuv420p', '-an', out],
    check=True)
print(f'wrote {out}')
