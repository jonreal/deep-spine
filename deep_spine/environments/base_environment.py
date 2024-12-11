import cv2
import os
import numpy as np
import mujoco
from mujoco.glfw import glfw
from typing import Optional
from PIL import Image
import OpenGL.GL as gl


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
        local_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = os.path.join(local_dir, xml_path)
        self.frame_rate = frame_rate
        self._initialize_mujoco()

        if self.frame_rate:
            self._initialize_render(azimuth, elevation, distance, lookat)
        if save_frames_to_path and self.frame_rate:
            self.save_frames_to_path = save_frames_to_path
            self._initialize_video_writer()
            #os.makedirs(save_frames_to_path, exist_ok=True)
            self.frame_count = 0
        else:
            self.save_frames_to_path = None

    def _initialize_video_writer(self):
        width, height = glfw.get_framebuffer_size(self.window)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
        os.makedirs(os.path.dirname(self.save_frames_to_path), exist_ok=True)
        self.video_writer = cv2.VideoWriter(
            f"{self.save_frames_to_path}.mp4",
            fourcc, 100.0, (width, height)
        )


    def _initialize_mujoco(self):
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def _initialize_render(self, azimuth, elevation, distance, lookat):
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
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
        glfw.set_window_pos(self.window, 2000, -2000)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window
        )
        self.viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    def _render(self):
        if (self.frame_rate
            and (
                self.data.time % (1/self.frame_rate)
                < (1/self.simulation_sampling_rate)
            )
        ):

            mujoco.mjv_updateScene(
                  self.model, self.data, self.opt, None, self.cam,
                  mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
            )
            mujoco.mjr_render(self.viewport, self.scene, self.context)

           # Add timestamp text
            epsstamp = f"Eps: {int(self.eps)}"
            timestamp = f"Time: {self.data.time:.2f}s"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, 
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, 
                self.viewport, 
                epsstamp,
                timestamp,
                self.context
            )

            if self.save_frames_to_path:
                width, height = glfw.get_framebuffer_size(self.window)
                image = np.zeros((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(image, None, self.viewport, self.context)
                image = cv2.cvtColor(np.flip(image, axis=0), cv2.COLOR_RGB2BGR)
                self.video_writer.write(image)
                self.frame_count += 1

            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def close(self):
        self.data = None
        self.model = None
        if self.frame_rate:
            if self.window:
                glfw.set_window_should_close(self.window, True)
                glfw.destroy_window(self.window)
            self.context.free()
            glfw.terminate()
            self.cam = None
            self.scene = None
            self.context = None
            self.viewport = None
        self.video_writer.release()
