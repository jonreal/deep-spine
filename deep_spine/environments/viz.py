import numpy as np
import mujoco

class HeatmapGeom:
    def __init__(
        self,
        size: int,
        center: np.ndarray,
        W: np.ndarray,
        xlabel = None,
        ylabel = None
    ):
        self.size = size
        self.center = center
        self.W = W
        self.n = W.shape[0]
        self.xlabel = xlabel
        self.ylabel = ylabel

    def draw(self, scene):
        for i in range(self.n):
            for j in range(self.n):
                w = self.W[i][j]
                if w < 0.5:
                    # scale 0.0->1.0 from w in [0,0.5]
                    scale = w / 0.5
                    R = 0.5*scale + 0*(1-scale)  # from 0 to 0.5
                    G = 0.5*scale + 0*(1-scale)  # from 0 to 0.5
                    B = 0.5*scale + 1*(1-scale)  # from 1 to 0.5
                else:
                    # scale 0.0->1.0 from w in [0.5,1.0]
                    scale = (w - 0.5) / 0.5
                    R = 1*scale + 0.5*(1-scale)  # from 0.5 to 1
                    G = 0*(scale) + 0.5*(1-scale) # from 0.5 to 0
                    B = 0*(scale) + 0.5*(1-scale) # from 0.5 to 0
                w_rgba = np.array([R, G, B, 0.1])
                center = self.center + np.array([0.085*i, 0.085*j, 0])
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

                tx_center = center + np.array([0.02, -0.025, 0])
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_LABEL,
                    size=self.size*np.array([1, 1, 0.01]),
                    pos=tx_center,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 0, 0])
                )
                scene.geoms[scene.ngeom].label = f"{w:.2f}"
                scene.ngeom += 1

                if i == 0:
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
                    if self.xlabel:
                        scene.geoms[scene.ngeom].label = f"{self.xlabel[j]}"
                    else:
                        scene.geoms[scene.ngeom].label = f"  {j}"
                    scene.ngeom += 1
                if j == 0:
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
                    tx_center = center + np.array([0.02, -0.17, 0])
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=self.size*np.array([1, 1, 0.01]),
                        pos=tx_center,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 0])
                    )
                    if self.ylabel:
                        scene.geoms[scene.ngeom].label = f"{self.ylabel[i]}"
                    else:
                        scene.geoms[scene.ngeom].label = f"{i}"
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
        add_zero_line = False
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
        self.source = source
        for i in range(ntraces):
            self.figure.linepnt[i] = window
            self.figure.linedata[i][:2*window:2] = np.linspace(0, 1, window)
            self.figure.linedata[i][1:2*window:2] = self.buffer[i,:]
        if add_zero_line:
            self.figure.linepnt[ntraces] = 2
            self.figure.linedata[ntraces][:4:2] = np.array([0, 1])
            self.figure.linedata[ntraces][1:4:2] = np.array([0, 0])
            self.figure.linergb[ntraces][:] = np.array([1, 1, 1])


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
        title: str = None
    ):
        self.ylabel = None
        self.nbars = nbars
        self.viewport = viewport
        self.figure = mujoco.MjvFigure()
        self.figure.figurergba = np.array([0, 0, 0, 0])
        self.figure.panergba = np.array([0, 0, 0, 0])
        self.figure.flg_ticklabel[1] = 0
        self.figure.linewidth = 10
        self.figure.minwidth = '0000000'
        self.figure.title = title
        self.figure.yformat = '%.2f'
        self.source = source
        for i in range(nbars):
            self.figure.linepnt[i] = 2
            self.figure.linedata[i][:4:2] = np.array([0, self.source[i]])
            self.figure.linedata[i][1:4:2] = np.array([i, i])
            self.figure.range[0][0] = 0
            self.figure.range[0][1] = 1
        if ylim is not None:
            self.figure.range[1][0] = -1
            self.figure.range[1][1] = nbars

    def update(self):
        for i in range(self.nbars):
            self.figure.linedata[i][:4:2] = np.array([0, self.source[i]])


