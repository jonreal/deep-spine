"""Export musculoskeletal plant data for the SI plant figure (pubplots_si_6.m).

Queries the arm26 MuJoCo model directly, so every panel reflects the model as
simulated:
  - muscle paths (via / wrap points) at the rest posture, plus segment endpoints
  - moment arms and normalized fiber length vs joint angle (shoulder sweep with
    the elbow at 0, and elbow sweep with the shoulder at 0)
  - force-length (active, passive) and force-velocity curves from MuJoCo's own
    muscle functions (mju_muscleGain / mju_muscleBias) with the model's
    parameters, normalized by peak force F0

Usage (repo root, venv active):
    python analysis/si_plant_export.py
writes analysis/si_plant.mat
"""
import os
import numpy as np
import mujoco
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'deep_spine', 'environments', 'arm26.xml')
OUT = os.path.join(HERE, 'si_plant.mat')

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]

gp = m.actuator_gainprm          # [range0 range1 force scale lmin lmax vmax fpmax fvmax]
lr = m.actuator_lengthrange
L0 = (lr[:, 1] - lr[:, 0]) / (gp[:, 1] - gp[:, 0])   # optimal fiber length


def norm_length(length):
    """Actuator length -> fiber length in units of L0 (MuJoCo convention)."""
    return gp[:, 0] + (length - lr[:, 0]) / L0


def moment_arms():
    """nu x nv moment-arm matrix at the current state (d length / d q)."""
    M = np.zeros((m.nu, m.nv))
    for i in range(m.nu):
        n, a = d.moment_rownnz[i], d.moment_rowadr[i]
        M[i, d.moment_colind[a:a + n]] = d.actuator_moment[a:a + n]
    return M


# ---- geometry at rest ----------------------------------------------------
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
# wrap_xpos stores two 3-D points per row; ten_wrapadr indexes points
wrap_pts = d.wrap_xpos.reshape(-1, 3)
paths = np.empty((m.ntendon,), dtype=object)
for t in range(m.ntendon):
    a, n = d.ten_wrapadr[t], d.ten_wrapnum[t]
    paths[t] = wrap_pts[a:a + n, :2].copy()              # planar (x, y)
joints = d.xpos[1:, :2].copy()                            # shoulder, elbow, wrist bodies
cyl = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
cyl_xy = d.geom_xpos[cyl, :2].copy()
cyl_r = m.geom_size[cyl, 0].copy()
body_mass = m.body_mass.copy()

# ---- joint-angle sweeps --------------------------------------------------
q = np.radians(np.linspace(-120, 120, 241))
MA = np.zeros((2, len(q), m.nu))     # [joint swept, angle, muscle] moment arm on that joint
LN = np.zeros((2, len(q), m.nu))     # normalized fiber length
for jnt in range(2):
    for k, qk in enumerate(q):
        mujoco.mj_resetData(m, d)
        d.qpos[jnt] = qk
        mujoco.mj_forward(m, d)
        MA[jnt, k, :] = moment_arms()[:, jnt]
        LN[jnt, k, :] = norm_length(d.actuator_length.copy())

# ---- force-length-velocity (muscle 0; all muscles share the curve shape) --
i0 = 0
prm = gp[i0, :9].copy()
F0 = prm[2]
lt = np.linspace(gp[i0, 4], gp[i0, 5], 400)              # fiber length, L0 units
vt = np.linspace(-1, 1, 401)                              # velocity, units of L0*vmax
vmax = prm[6]
length_of = lambda l: lr[i0, 0] + (l - prm[0]) * L0[i0]
vel_of = lambda v: v * L0[i0] * vmax
acc0 = m.actuator_acc0[i0]

FL_active = np.array([-mujoco.mju_muscleGain(length_of(l), 0.0, lr[i0], acc0, prm) / F0 for l in lt])
FL_passive = np.array([-mujoco.mju_muscleBias(length_of(l), lr[i0], acc0, prm) / F0 for l in lt])
FV = np.array([-mujoco.mju_muscleGain(length_of(1.0), vel_of(v), lr[i0], acc0, prm) / F0 for v in vt])

savemat(OUT, {
    'names': np.array(names, dtype=object),
    'paths': paths,
    'joints': joints,
    'cyl_xy': cyl_xy, 'cyl_r': cyl_r,
    'body_mass': body_mass,
    'q_deg': np.degrees(q),
    'moment_arm': MA,
    'norm_length': LN,
    'op_range': gp[0, :2],
    'L0': L0,
    'lt': lt, 'FL_active': FL_active, 'FL_passive': FL_passive,
    'vt': vt, 'FV': FV,
    'F0_ref': F0,
})
print(f'wrote {OUT}')
print('FL_active peak', FL_active.max().round(3), 'at', lt[FL_active.argmax()].round(3))
print('FV at v=0', FV[len(vt) // 2].round(3), ' v=-1', FV[0].round(3), ' v=+1', FV[-1].round(3))
