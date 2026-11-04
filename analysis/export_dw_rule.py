"""Export the per-step Hebbian weight update computed by the learning rule.

W is stored as float32 in the simulation, so per-step increments (~1e-7) are
only a few float32 ulps and differencing the logged W gives a quantized
staircase. This replays HebbianLearner.step() from the logged signals
(env.txt: act_Ia = pre, act_MN = post; hebbian.txt: W(t-1)) and writes the
update each step applied BEFORE it was rounded into W. It is the same run, not
a re-simulation: the float32 path of the replay reproduces every logged W(t)
bit-for-bit, and the number of mismatches is reported.

Output: <run>/dw_rule.bin, float32, same layout as hebbian.txt (one row of
W.flatten() per step, row-major (i,j) -> i*6+j; row t = update applied at
step t, so W(t) = f32(W(t-1) + dW(t))). Units: weight per step.

Usage (repo root, venv active):
    python analysis/export_dw_rule.py <run_dir>
"""
import json
import sys

import numpy as np
from scipy import signal

run = sys.argv[1]
NC, N = 59, 6
C_IA = 1 + 2 + 2 + 6*4                    # env.txt column offsets (ds_load_trial)
C_MN = C_IA + 6 + 6
CHUNK = 1_000_000

cfg = json.load(open(f'{run}/config.json'))['hebbian_kwargs'][0]
gain, lr = cfg['gain'], cfg['learning_rate']
Km, Kl = cfg['Kmean'], cfg['Klearn']
sf = cfg['synaptic_filter']
b_hp, a_hp = signal.butter(sf['order'], sf['cutoff'], sf['type'])
b_m, a_m = np.array([0, Km]), np.array([1, Km-1])
b_l, a_l = np.array([0, Kl]), np.array([1, Kl-1])

env = np.memmap(f'{run}/env.txt', dtype=np.float64, mode='r').reshape(-1, NC)
heb = np.memmap(f'{run}/hebbian.txt', dtype=np.float32, mode='r').reshape(-1, N*N)
H = heb.shape[0]                          # env row t <-> hebbian row t
print(f'env rows {env.shape[0]}, hebbian rows {H}')

zi_hp = np.zeros((1, N)); zi_m = np.zeros((1, N)); zi_l = np.zeros((1, N*N))
n_bad = n_moved = 0
with open(f'{run}/dw_rule.bin', 'wb') as out:
    for c0 in range(0, H, CHUNK):
        c1 = min(H, c0 + CHUNK)
        post = np.array(env[c0:c1, C_MN:C_MN+N])
        pre = np.array(env[c0:c1, C_IA:C_IA+N])
        hp, zi_hp = signal.lfilter(b_hp, a_hp, post, axis=0, zi=zi_hp)
        mp, zi_m = signal.lfilter(b_m, a_m, post, axis=0, zi=zi_m)
        pdev = gain*hp - mp
        # W(t-1); the update at row 0 used the initial weights, which are not
        # logged, so row 0 of hebbian.txt stands in (its update is ~1e-10).
        Wprev = np.array(heb[max(c0-1, 0):c1-1] if c0 > 0 else
                         np.vstack([heb[0:1], heb[0:c1-1]]))
        raw = (pdev[:, :, None] * pre[:, None, :]).reshape(-1, N*N) * Wprev
        ls, zi_l = signal.lfilter(b_l, a_l, raw, axis=0, zi=zi_l)
        rate = np.repeat(mp**4 * lr, N, axis=1)            # per-row rate, (n, 36)
        comp = np.where(ls >= 0, 1 - Wprev, Wprev)
        dW = comp * ls * rate
        # float32 path, as the learner computes it, to verify against the log
        ls32 = ls.astype(np.float32); W32 = Wprev.astype(np.float32)
        comp32 = np.where(ls32 >= 0, np.float32(1) - W32, W32)
        dW32 = comp32 * ls32 * (mp.astype(np.float32)**4 * np.float32(lr)).repeat(N, axis=1)
        Wpred = (W32 + dW32.astype(np.float32)).astype(np.float32)
        Wlog = np.array(heb[c0:c1])
        chk = slice(1 if c0 == 0 else 0, None)             # row 0: W(-1) unknown
        n_bad += np.count_nonzero(Wpred[chk] != Wlog[chk])
        n_moved += np.count_nonzero(Wlog[chk] != W32[chk])
        out.write(dW.astype(np.float32).tobytes())
        print(f'  {100*c1/H:5.1f}%  mismatches so far {n_bad}', end='\r')
print(f'\nwrote {run}/dw_rule.bin ({H} x {N*N} float32)')
print(f'replay vs logged W: {n_bad} mismatching entries of {(H-1)*N*N} '
      f'({n_moved} entries changed)')
