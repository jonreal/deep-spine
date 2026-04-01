import numpy as np
import mujoco
import colorsys

class Annotation:
    def __init__(
        self,
        text: str,
        viewport: mujoco.MjrRect,
        gridpos: int = 0,
        font: int =0
    ):
        self.text = text
        self.viewport = viewport
        self.gridpos = gridpos
        self.font = font


class HeatmapGeom:
    PLANE_MAP = {
        "xy": (0, 1, 2),
        "xz": (0, 2, 1),
        "yz": (1, 2, 0),
    }

    def __init__(
        self,
        size: int,
        center: np.ndarray,
        W: np.ndarray,
        xlabel = None,
        ylabel = None,
        title = None,
        notext = False,
        W0 = None,
        min_weight = -1,
        max_weight = 1,
        spacing = 2.15,
        text_color = np.array([0, 0, 0, 0]),
        plane = "xy"
    ):
        self.size = size
        self.center = center
        self.W = W
        self.nrows = W.shape[0]
        self.ncols = W.shape[1]
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.notext = notext
        self.W0 = W0
        self.spacing = spacing
        self.text_color = text_color
        self.row_ax, self.col_ax, self.thin_ax = self.PLANE_MAP[plane]

    def _offset(self, row_val, col_val, thin_val=0):
        v = np.zeros(3)
        v[self.row_ax] = row_val
        v[self.col_ax] = col_val
        v[self.thin_ax] = thin_val
        return v

    def draw(self, scene):

        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.W0 is not None:
                    current_w = self.W[i][j]
                    initial_w = self.W0[i][j]
                    if current_w < initial_w:

                        # Calculate how far it decreased as a proportion of
                        # maximum possible decrease 0 = no change, 1 = maximum
                        # decrease (to min_weight)

                        max_decrease = initial_w - self.min_weight
                        if max_decrease > 0:
                            decrease_proportion = (initial_w - current_w) / max_decrease
                        else:
                            decrease_proportion = 0

                        # Blue (maximum decrease) to white (no change)
                        R = 1 - decrease_proportion
                        G = 1 - decrease_proportion
                        B = 1  # Always has blue component

                    # If weight stayed the same or increased from initial value
                    else:

                        # Calculate how far it increased as a proportion of
                        # maximum possible increase 0 = no change, 1 = maximum
                        # increase (to max_weight)

                        max_increase = self.max_weight - initial_w
                        if max_increase > 0:
                            increase_proportion = (
                                (current_w - initial_w) / max_increase
                            )
                        else:
                            increase_proportion = 0

                        # White (no change) to red (maximum increase)
                        R = 1  # Always has red component
                        G = 1 - increase_proportion
                        B = 1 - increase_proportion

                else:
                    # Your existing code for when W0 is not supplied
                    w = self.W[i][j]
                    if w <= 0.5:
                        # Map [0, 0.5] to [blue, white]
                        scale = w / 0.5
                        R = scale  # From 0 to 1
                        G = scale  # From 0 to 1
                        B = 1  # Always blue
                    else:
                        # Map [0.5, 1] to [white, red]
                        scale = (w - 0.5) / 0.5
                        R = 1  # Always red
                        G = 1 - scale  # From 1 to 0
                        B = 1 - scale  # From 1 to 0

                w_rgba = np.array([R, G, B, 0.15])

                center = (
                    self.center
                    + self._offset(
                        self.spacing*self.size*i,
                        self.spacing*self.size*j)
                )
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=self.size*self._offset(1, 1, 0.01),
                    pos=center,
                    mat=np.eye(3).flatten(),
                    rgba=w_rgba
                )
                scene.geoms[scene.ngeom].emission = 1.0
                scene.geoms[scene.ngeom].specular = 0.0
                scene.geoms[scene.ngeom].shininess = 0.0
                scene.ngeom += 1

                if not self.notext:
                    tx_center = center + self._offset(0.02, -0.025)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*self._offset(1, 1, 0.01),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    scene.geoms[scene.ngeom].label = f"{self.W[i][j]:.2f}"
                    scene.ngeom += 1

                if i == 0 and self.xlabel is not None:
                    tx_center = center + self._offset(-0.04, -0.025)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*self._offset(1, 1, 0.01),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=self.text_color
                    )
                    scene.geoms[scene.ngeom].label = "  |"
                    scene.ngeom += 1
                    tx_center = center + self._offset(-0.08, -0.025)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*self._offset(1, 1, 0.01),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=self.text_color
                    )
                    scene.geoms[scene.ngeom].label = f"{self.xlabel[j]}"
                    scene.ngeom += 1
                if j == 0 and self.ylabel is not None:
                    tx_center = center + self._offset(0.01, -0.06)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*self._offset(1, 1, 0.01),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=self.text_color
                    )
                    scene.geoms[scene.ngeom].label = "_"
                    scene.ngeom += 1
                    tx_center = center + self._offset(0.02, -0.2)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*self._offset(1, 1, 0.01),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=self.text_color
                    )
                    scene.geoms[scene.ngeom].label = f"{self.ylabel[i]}"
                    scene.ngeom += 1
        if self.title is not None:
            step = self.spacing * self.size
            tx_center = self.center + self._offset(
                self.nrows * step + step,
                (self.ncols - 1) * step / 2 - step
            )
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=self.size*self._offset(1, 1, 0.01),
                pos=tx_center,
                mat=np.eye(3).flatten(),
                rgba=self.text_color
            )
            scene.geoms[scene.ngeom].label = self.title
            scene.ngeom += 1

class TracesFigure:
    def __init__(
        self,
        viewport: mujoco.MjrRect,
        ntraces: int,
        window: int,
        source: np.ndarray = None,
        ylabel: str=None,
        xlim: np.ndarray = None,
        ylim: np.ndarray = None,
        yformat: str = '%.2f',
        ysymmetric: bool = False,
        gridsize: np.ndarray = None,
        text_color = np.array([1, 1, 1, 1])
    ):
        self.ylabel = ylabel
        self.text_color = text_color
        self.ntraces = ntraces
        self.buffer = np.zeros((ntraces, window))
        self.window = window
        self.viewport = viewport
        self.figure = mujoco.MjvFigure()
        self.figure.figurergba = np.array([0, 0, 0, 0])
        self.figure.panergba = np.array([0, 0, 0, 0])
        self.figure.flg_ticklabel[0] = 0
        self.figure.flg_extend = 1
        self.figure.linewidth = 1
        self.figure.minwidth = '0000000'
        self.figure.yformat = yformat
        if ysymmetric:
            self.figure.flg_symmetric = 1
            self.figure.gridsize[1] = 3
        if gridsize is not None:
            self.figure.gridsize = gridsize

        self.source = source
        for i in range(ntraces):
            self.figure.linepnt[i] = window
            self.figure.linedata[i][:2*window:2] = np.linspace(0, 1, window)
            self.figure.linedata[i][1:2*window:2] = self.buffer[i,:]

        if xlim is not None:
            self.figure.range[0][0] = xlim[0]
            self.figure.range[0][1] = xlim[1]
        if ylim is not None:
            self.figure.range[1][0] = ylim[0]
            self.figure.range[1][1] = ylim[1]

    def update(self):
        if self.source is not None:
            source = self.source.flatten()
            self.buffer[:,:-1] = self.buffer[:,1:]
            self.buffer[:,-1] = source
            for i in range(self.ntraces):
                self.figure.linedata[i][1:2*self.window:2] = self.buffer[i,:]

class BarsFigure:
    def __init__(
        self,
        viewport: mujoco.MjrRect,
        nbars: int,
        source: np.ndarray = None,
        xlim: np.ndarray = None,
        ylim: np.ndarray = None,
        title: str = None,
        xformat: str = '%.2f',
        gridsize: np.ndarray = None,
        ysymmetric: bool = False,
    ):
        self.title = title
        self.ylabel = None
        self.nbars = nbars
        self.viewport = viewport
        self.figure = mujoco.MjvFigure()
        self.figure.flg_barplot = 1
        self.figure.figurergba = np.array([0, 0, 0, 0])
        self.figure.panergba = np.array([0, 0, 0, 0])
        self.figure.linewidth = 10
        self.figure.minwidth = '0'
        self.figure.xformat = xformat
        self.source = source
        for i in range(nbars):
            self.figure.linepnt[i] = 2
            self.figure.linedata[i][:4:2] = np.array([0, self.source[i]])
            self.figure.linedata[i][1:4:2] = np.array([i, i])

        if ysymmetric:
            self.figure.flg_symmetric = 1

        if xlim is not None:
            self.figure.range[0][0] = xlim[0]
            self.figure.range[0][1] = xlim[1]
        if ylim is not None:
            self.figure.range[1][0] = -1
            self.figure.range[1][1] = nbars
        if gridsize is not None:
            self.figure.gridsize[0] = gridsize[0]
            self.figure.gridsize[1] = gridsize[1]

        self.figure.flg_ticklabel[1] = 0
        self.figure.flg_ticklabel[0] = 1

    def update(self):
        for i in range(self.nbars):
            self.figure.linedata[i][:4:2] = np.array([0, self.source[i]])
        self.figure.selection = self.source[0]
        self.figure.title = f'{self.title}, {self.source[0]:.2f}'


def make_spine_viz(data, spine, supraspine, xs, ys, H, W, T, timestep):
    nq = data.qpos.shape[0]
    traces = [
        ('pos', data.qpos, nq, [-0.25, 0.25], True),
        ('qvel', data.qvel, nq, [-0.5, 0.5], False),
        ('len', spine.length, spine.n, [0, 1], False),
        ('vel', spine.velocity, spine.n, [-0.1, 0.1], True),
        ('force', spine.force_signed, spine.n, [-100, 100], True),
        ('drive', spine.drive, spine.n, [0, 1], False),
        ('act_II', spine.act_II, spine.n, [0, 1], False),
        ('act_Ia', spine.act_Ia, spine.n, [0, 1], False),
        ('MN', spine.act_MN, spine.n, [0, 1], False),
        ('act_MN', spine.act_muscle, spine.n, [0, 1], False),
        ('act', spine.act, spine.n, [0, 1], False),
    ]
    N = int(T / timestep)
    figs = []
    for i, (label, source, n, ylim, ysymmetric) in enumerate(traces):
        figs.append(
            TracesFigure(
                viewport=mujoco.MjrRect(xs, ys - (i+1)*H, W, H),
                ntraces=n,
                window=N,
                source=source,
                ylabel=label,
                xlim=np.array([0, T]),
                ylim=ylim,
                yformat='%.2f',
                ysymmetric=ysymmetric,
                gridsize=np.array([2, 3])
            )
        )
    #xlabel
    fig = TracesFigure(
        viewport=mujoco.MjrRect(xs, ys-11*H - 20, W, 10),
        ntraces=0,
        window=N,
        xlim=np.array([-T, 0])
    )
    fig.figure.yformat = ''
    fig.figure.flg_ticklabel[0] = 1
    fig.figure.flg_ticklabel[1] = 1
    figs.append(fig)
    return figs


make_pendulum_viz = make_spine_viz

def make_hebbian_viz(learner, xs, ys, H, W, T, timestep, legend=False,
                     showlabel=True):
    n_rows = len(learner.learning_signal)
    traces = [
        ('pre', learner.pre_synaptic, learner.pre_synaptic.shape[0],
            [0, 1], False),
        ('post*', learner.post_dev, learner.post_dev.shape[0],
            [-0.5, 0.25], False),
    ]
    for r in range(n_rows):
        traces.append(
            (f'L_{r+1}j', learner.learning_signal[r][:],
                learner.learning_signal[r][:].shape[0],
                [-0.001, 0.001], False)
        )
    N = int(T / timestep)
    figs = []
    for i, (label, source, n, ylim, ysymmetric) in enumerate(traces):
        if not showlabel:
            label = None
        figs.append(
            TracesFigure(
                viewport=mujoco.MjrRect(xs, ys - (i+1)*H, W, H),
                ntraces=n,
                window=N,
                source=source,
                ylabel=label,
                xlim=np.array([0, T]),
                ylim=ylim,
                yformat='%.3f',
                ysymmetric=ysymmetric,
                gridsize=np.array([2, 3])
            )
        )

    if legend:
        fig = TracesFigure(
            viewport=mujoco.MjrRect(
                xs+W, ys -int(2.3*H), int(W/3.5), int(2.3*H)
            ),
            ntraces=learner.learning_signal[-1][:].shape[0],
            window=N,
            source=None,
            xlim=np.array([0, T]),
            ylim=np.array([-20, -1]),
            yformat='%.5f'
        )
        figs.append(fig)

    n_panels = len(traces)
    fig = TracesFigure(
        viewport=mujoco.MjrRect(xs, ys - n_panels*H - 90, W, 10),
        ntraces=0,
        window=N,
        xlim=np.array([-T, 0])
    )
    figs.append(fig)

    figs[-1].figure.yformat = ''
    figs[-1].figure.flg_ticklabel[0] = 1
    figs[-1].figure.flg_ticklabel[1] = 1

    if legend:
        figs[-2].figure.flg_legend = 1
        figs[-2].figure.linename[0] = 'SF'
        figs[-2].figure.linename[1] = 'SE'
        figs[-2].figure.linename[2] = 'EF'
        figs[-2].figure.linename[3] = 'EE'
        figs[-2].figure.linename[4] = 'BF'
        figs[-2].figure.linename[5] = 'BE'
        figs[-2].figure.flg_ticklabel[0] = 0
        figs[-2].figure.flg_ticklabel[1] = 0
        figs[-2].figure.flg_extend = 0
    #figs[1].figure.flg_extend = 0

    return figs


def make_supraspine_viz(supraspine, xs, ys, H, W, T, timestep):
    n = supraspine.n
    traces = [
        ('act', supraspine.action, n, [0, 1], False),
        ('pro', supraspine.proprio, n, [0, 1], False),
        ('pos', supraspine.visual[0:1], 1, [-50*np.pi/180, 50*np.pi/180], True),
        ('vel', supraspine.visual[1:2], 1, [-5*np.pi, 5*np.pi], True),
    ]
    N = int(T / timestep)
    figs = []
    for i, (label, source, ntraces, ylim, ysymmetric) in enumerate(traces):
        figs.append(
            TracesFigure(
                viewport=mujoco.MjrRect(xs, ys - (i+1)*H, W, H),
                ntraces=ntraces,
                window=N,
                source=source,
                ylabel=label,
                xlim=np.array([0, T]),
                ylim=ylim,
                yformat='%.2f',
                ysymmetric=ysymmetric,
                gridsize=np.array([2, 3])
            )
        )
    # xlabel
    fig = TracesFigure(
        viewport=mujoco.MjrRect(xs, ys - (len(traces)+1)*H - 20, W, 10),
        ntraces=0,
        window=N,
        xlim=np.array([-T, 0])
    )
    fig.figure.yformat = ''
    fig.figure.flg_ticklabel[0] = 1
    fig.figure.flg_ticklabel[1] = 1
    figs.append(fig)
    return figs


def make_bars_viz(source, xs, ys, W, H, title=None, nbars=1,
                  xlim=None, ylim=None, gridsize=None, ysymmetric=False,
                  show_selection=False):
    """Factory for BarsFigure."""
    xlim = np.array(xlim) if xlim else np.array([0, 1])
    ylim = np.array(ylim) if ylim else np.array([-1, 1])
    gridsize = np.array(gridsize) if gridsize else np.array([5, 2])

    fig = BarsFigure(
        viewport=mujoco.MjrRect(xs, ys, W, H),
        nbars=nbars,
        title=title,
        source=source,
        xlim=xlim,
        ylim=ylim,
        gridsize=gridsize,
        ysymmetric=ysymmetric
    )
    if show_selection:
        fig.figure.flg_selection = 1
        fig.figure.selection = source.item()
    return [fig]


def make_heatmap_geom(W, lookat, center_offset, size,
                      min_weight=-1, max_weight=1,
                      xlabel=None, ylabel=None, title=None,
                      notext=False, clone_W0=False, spacing=2.15,
                      plane="xy"):
    """Factory for HeatmapGeom."""
    center = lookat + np.array(center_offset)
    W0 = W.clone().detach() if clone_W0 else None

    geom = HeatmapGeom(
        center=center,
        size=size,
        W=W,
        W0=W0,
        min_weight=min_weight,
        max_weight=max_weight,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        notext=notext,
        spacing=spacing,
        plane=plane
    )
    return [geom]


def build_tb_tag(config):
    """Build a short tag for TensorBoard directory names.
    Extracts only leaf key=value pairs, no hierarchy prefix."""
    if config is None:
        return ""
    leaves = []
    _collect_leaves(config, leaves)
    return "_".join(format_value(v) for _, v in leaves)

def _is_primitive_list(obj):
    """True if obj is a list of only scalars (no nested dicts/lists)."""
    return isinstance(obj, list) and all(
        isinstance(x, (int, float, str)) for x in obj
    )

def _collect_leaves(obj, leaves):
    """Recursively collect (leaf_key, value) pairs from nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            if _is_primitive_list(v):
                leaves.append((k, v))
            elif isinstance(v, (dict, list)):
                _collect_leaves(v, leaves)
            else:
                leaves.append((k, v))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _collect_leaves(item, leaves)


def format_value(v):
    """Format a value for annotation display."""
    if isinstance(v, (int, float)):
        return f'{v:.3g}'
    elif isinstance(v, str):
        return v[:12]
    elif isinstance(v, list):
        seen = set()
        compact = []
        for x in v:
            if isinstance(x, (int, float)) and x == 0:
                continue
            fv = format_value(x)
            if fv not in seen:
                seen.add(fv)
                compact.append(fv)
        return "x".join(compact) if compact else "0"
    else:
        return str(v)


def build_annotation_text(config, prefix=""):
    """Build annotation text from a config dict."""
    if config is None:
        return ""
    lines = []
    # Handle dictionaries
    if isinstance(config, dict):
        for k, v in sorted(config.items()):
            k_prefix = f"{prefix}_{k[:3]}" if prefix else k[:3]
            if isinstance(v, dict):
                # Recursive call for nested dictionaries
                nested_lines = build_annotation_text(v, k_prefix)
                lines.extend(nested_lines.split("\n"))
            elif isinstance(v, list):
                # Handle list values
                for i, item in enumerate(v):
                    if isinstance(item, (dict, list)):
                        # Use index in the prefix for nested structures
                        item_prefix = f"{k_prefix}_{i}"
                        nested_lines = build_annotation_text(item, item_prefix)
                        lines.extend(nested_lines.split("\n"))
                    else:
                        # Simple list items
                        lines.append(f"{k_prefix}[{i}]={format_value(item)}")
            else:
                # Simple key-value pairs
                lines.append(f"{k_prefix}={format_value(v)}")
    # Handle lists
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, (dict, list)):
                # Use index in the prefix for nested structures
                item_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                nested_lines = build_annotation_text(item, item_prefix)
                lines.extend(nested_lines.split("\n"))
            else:
                # Simple list items
                index_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                lines.append(f"{index_prefix}={format_value(item)}")
    # Handle simple types
    else:
        lines.append(f"{prefix}={format_value(config)}")
    return "\n".join(line for line in lines if line)
