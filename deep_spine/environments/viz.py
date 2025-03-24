import numpy as np
import mujoco

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
    def __init__(
        self,
        size: int,
        center: np.ndarray,
        W: np.ndarray,
        xlabel = None,
        ylabel = None,
        notext = False,
        W0 = None,
        min_weight = -1,
        max_weight = 1,
        spacing = 2.15
    ):
        self.size = size
        self.center = center
        self.W = W
        self.n = W.shape[0]
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.notext = notext
        self.W0 = W0
        self.spacing = spacing

    def draw(self, scene):
        for i in range(self.n):
            for j in range(self.n):
                if self.W0 is not None:
                    delta_W = self.W[i][j] - self.W0[i][j]

                    W_min_dev = self.min_weight - self.W0[i][j]
                    W_max_dev = self.max_weight - self.W0[i][j]

                    if delta_W < 0:
                        scale = (delta_W - W_min_dev) / (-W_min_dev)
                    else:
                        scale = delta_W / W_max_dev

                else:
                    scale = (
                        (self.W[i][j] - self.min_weight) /
                        (self.max_weight - self.min_weight)
                    )

                R = 1 if scale > 0.5 else 2 * scale
                G = 2 * scale if scale < 0.5 else 2 * (1 - scale)
                B = 1 if scale < 0.5 else 2 * (1 - scale)

                w_rgba = np.array([R, G, B, 0.1])

            #        w = self.W[i][j] - self.W0[i][j]  # Compute raw w
            #        if w < 0:
            #            scale = w / -self.W0[i][j]
            #            R = 1 * (1 - scale)  # From 0 to 1
            #            G = 1 * (1 - scale)  # From 0 to 1
            #            B = 1  # Always blue
            #        else:
            #            scale = w / (1 - self.W0[i][j])
            #            R = 1  # Always red
            #            G = 1 * (1 - scale)  # From 1 to 0
            #            B = 1 * (1 - scale)  # From 1 to 0
            #        w_rgba = np.array([R, G, B, 0.1])
            #    else:
            #        w = self.W[i][j]
            #        if w <= 0.5:
            #            # Map [0, 0.5] to [blue, white]
            #            scale = w / 0.5
            #            R = 1 * scale  # From 0 to 1
            #            G = 1 * scale  # From 0 to 1
            #            B = 1  # Always blue
            #        else:
            #            # Map [0.5, 1] to [white, red]
            #            scale = (w - 0.5) / 0.5
            #            R = 1  # Always red
            #            G = 1 * (1 - scale)  # From 1 to 0
            #            B = 1 * (1 - scale)  # From 1 to 0

            #    w_rgba = np.array([R, G, B, 0.1])

                center = (
                    self.center
                    + np.array(
                        [self.spacing*self.size*i,
                         self.spacing*self.size*j, 0]
                    )
                )
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=self.size*np.array([1, 1, 0.01]),
                    pos=center,
                    mat=np.eye(3).flatten(),
                    rgba=w_rgba
                )
                scene.geoms[scene.ngeom].emission = 1.0
                scene.geoms[scene.ngeom].specular = 0.0
                scene.geoms[scene.ngeom].shininess = 0.0
                scene.ngeom += 1

                if not self.notext:
                    tx_center = center + np.array([0.02, -0.025, 0])
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*np.array([1, 1, 0.01]),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    scene.geoms[scene.ngeom].label = f"{self.W[i][j]:.2f}"
                    scene.ngeom += 1

                if i == 0 and self.xlabel is not None:
                    tx_center = center + np.array([-0.04, -0.025, 0])
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*np.array([1, 1, 0.01]),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    scene.geoms[scene.ngeom].label = "  |"
                    scene.ngeom += 1
                    tx_center = center + np.array([-0.08, -0.025, 0])
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*np.array([1, 1, 0.01]),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    scene.geoms[scene.ngeom].label = f"{self.xlabel[j]}"
                    scene.ngeom += 1
                if j == 0 and self.ylabel is not None:
                    tx_center = center + np.array([0.01, -0.06, 0])
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*np.array([1, 1, 0.01]),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    scene.geoms[scene.ngeom].label = "_"
                    scene.ngeom += 1
                    tx_center = center + np.array([0.02, -0.2, 0])
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*np.array([1, 1, 0.01]),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    scene.geoms[scene.ngeom].label = f"{self.ylabel[i]}"
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
        gridsize: np.ndarray = None
    ):
        self.ylabel = ylabel
        self.ntraces = ntraces
        self.buffer = np.zeros((ntraces, window))
        self.window = window
        self.viewport = viewport
        self.figure = mujoco.MjvFigure()
        self.figure.figurergba = np.array([0, 0, 0, 0])
        self.figure.panergba = np.array([0, 0, 0, 0])
        self.figure.flg_ticklabel[0] = 0
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

