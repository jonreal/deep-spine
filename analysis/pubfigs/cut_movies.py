"""Cut the real-time SI clips (Movies S2-S5) out of the canonical run's sim.mp4.
Movie S1, the time-lapse of the whole run, is made by timelapse_movie.py.

Usage (repo root, venv active):
    python analysis/pubfigs/cut_movies.py <path/to/sim.mp4> [outdir] [names]
Default outdir: analysis/pubfigs/out. Writes si-movie-start.mp4 / -early.mp4 / -late.mp4
/ -end.mp4; [names] is an optional comma list (e.g. end) to cut only those.

Frame mapping (frames are rendered after each physics step, so each episode's
t = 0 frame is skipped):
    frames per episode   FPE = EP*FR - 1
    frame f  ->  episode e = f // FPE,  time = EP*e + (f % FPE + 1)/FR   (s)
The video is written at VFPS = 10*FR (10x real time).

Each clip is WIN seconds long and time-locked to Fig. S3: the early and late
clips are centred on the Fig. S3 10 s windows (sample indices chosen by
figS3_timehistory.m at fs = 200 Hz); the start clip begins at t = 0. Output frame i
shows the simulated instant t0 + i/FR, using the most recent rendered frame at
or before it (sample-and-hold; the first frame is used for instants before the
first render). Clips are therefore exactly WIN*FR frames at a constant FR fps:
real-time playback (1 s simulated = 1 s played), with gaps at episode
boundaries filled by holding the previous frame. Resolution is kept native.
"""
import os
import subprocess
import sys

import numpy as np

FR   = 20                 # canonical render frame_rate (frames per simulated s)
EP   = 90.0               # episode length (s)
FPE  = int(EP*FR) - 1     # frames per episode in the video
VFPS = 10*FR              # container frame rate
FS   = 200                # log sampling rate (Hz)
WIN  = 20.0               # clip length (s)
CRF  = 18

# Fig. S3 10 s windows: start sample index (fs = 200). Clips are centred on
# them; 'start' is the beginning of the run.
SI_WIN = 10.0
NEP  = 2000               # episodes in the canonical run
wins = {
    'start': 0.0,
    'early': 2205201/FS + SI_WIN/2 - WIN/2,
    'late':  28600601/FS + SI_WIN/2 - WIN/2,
    'end':   NEP*EP - WIN,    # last WIN s of the run
}


def ftime(f):
    e, k = divmod(f, FPE)
    return EP*e + (k + 1)/FR


def first_frame_at_or_after(t):
    e = int(t // EP)
    k = max(1, -(-round((t - e*EP)*FR*1e6) // 10**6))   # ceil, robust to float noise
    if k > FPE:
        e, k = e + 1, 1
    return e*FPE + (k - 1)


def main():
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('analysis', 'pubfigs', 'out')
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else list(wins)
    os.makedirs(outdir, exist_ok=True)
    W, H = map(int, subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height', '-of', 'csv=p=0', src],
        check=True, capture_output=True, text=True).stdout.strip().split(','))
    nbytes = W*H*3
    nout = int(round(WIN*FR))
    for name, t0 in wins.items():
        if name not in only:
            continue
        t0 = max(0.0, t0)
        # output instants and, for each, the source frame shown (sample-and-hold)
        ti = t0 + np.arange(nout)/FR
        f_lo = max(0, first_frame_at_or_after(t0) - 1)
        f_hi = first_frame_at_or_after(ti[-1]) + 1
        fr_idx = np.arange(f_lo, f_hi + 1)
        fr_t = np.array([ftime(f) for f in fr_idx])
        pick = np.searchsorted(fr_t, ti + 1e-9, side='right') - 1
        pick = np.clip(pick, 0, len(fr_idx) - 1)
        # decode the needed source frames (raw RGB), then write the held sequence
        dec = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-ss', f'{f_lo/VFPS:.4f}', '-i', src,
             '-frames:v', str(len(fr_idx)), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
            stdout=subprocess.PIPE)
        frames = [dec.stdout.read(nbytes) for _ in range(len(fr_idx))]
        dec.wait()
        out = os.path.join(outdir, f'si-movie-{name}.mp4')
        enc = subprocess.Popen(
            ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{W}x{H}', '-r', str(FR), '-i', '-',
             '-c:v', 'libx264', '-crf', str(CRF), '-preset', 'slow',
             '-pix_fmt', 'yuv420p', '-an', out],
            stdin=subprocess.PIPE)
        for k in pick:
            enc.stdin.write(frames[k])
        enc.stdin.close()
        enc.wait()
        held = nout - len(np.unique(pick))
        print(f'wrote {out}: sim t = {t0:.3f}-{t0 + WIN:.3f} s ({t0/3600:.4f} h), '
              f'{nout} frames ({held} held)')


if __name__ == '__main__':
    main()
