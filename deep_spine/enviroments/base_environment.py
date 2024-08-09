import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
from typing import Optional
from PIL import Image


class BaseEnvironment(object):
    def __init__(
        self,
        xml_path: str = 'arm26.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        azimuth: int = 180,
        elevation: int = -90,
        distance: int = 2,
        lookat: np.ndarray = np.array([0.35, 0, 0.25]),
        ):
        self.xml_path = xml_path
        self.frame_rate = frame_rate
        self._initialize_mujoco()
        if self.frame_rate:
            self._initialize_render(azimuth, elevation, distance, lookat)
        if save_frames_to_path:
           self.save_frames_to_path = save_frames_to_path
           os.makedirs(save_frames_to_path, exist_ok=True)
           self.frame_count = 0
        else:
            self.save_frames_to_path = None

    def _initialize_mujoco(self):
        dirname = os.path.dirname(__file__)
        fpath = os.path.join(dirname,self.xml_path)
        self.model = mj.MjModel.from_xml_path(fpath)
        self.data = mj.MjData(self.model)

    def _initialize_render(self, azimuth, elevation, distance, lookat):
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.distance = distance
        self.cam.lookat = lookat
        glfw.init()
        self.window = glfw.create_window(
            1200,
            900,
            f'{self.type}-{self.task}',
            None,
            None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(
                self.model, mj.mjtFontScale.mjFONTSCALE_150.value
        )
        viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window
        )
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    def _render(self):
        if (self.frame_rate
            and int(self.data.time*self.frame_rate*10) % 10 == 0
        ):
            mj.mjv_updateScene(
                  self.model, self.data, self.opt, None, self.cam,
                  mj.mjtCatBit.mjCAT_ALL.value, self.scene
            )
            mj.mjr_render(self.viewport, self.scene, self.context)
            if self.save_frames_to_path:
                width, height = glfw.get_framebuffer_size(self.window)
                image = np.zeros((height, width, 3), dtype=np.uint8)
                mj.mjr_readPixels(image, None, self.viewport, self.context)
                image = np.flip(image, axis=0)
                self.frame_count += 1
                fpath = os.path.join(
                    self.save_frames_to_path,
                    f'frame{self.frame_count:05d}.png'
                )
                Image.fromarray(image).save(fpath)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def close(self):
        if self.frame_rate:
            if self.window:
                glfw.set_window_should_close(self.window, True)
                glfw.destroy_window(self.window)
            self.context.free()
            glfw.terminate()
        self.model = None
        self.data = None

        if self.frame_rate:
            self.cam = None
            self.scene = None
            self.context = None
            self.viewport = None
