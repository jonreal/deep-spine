import av
import os
import numpy as np
import mujoco
from mujoco.glfw import glfw
from typing import Optional
from PIL import Image
import OpenGL.GL as gl
import time

import subprocess


class BaseEnvironment(object):
    def __init__(
        self,
        xml_path: str = 'arm26.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        render_in_background: bool = False,
        render_lossless: bool = False,
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
            self.render_lossless = render_lossless
            self._initialize_render(azimuth, elevation, distance, lookat)
            if render_in_background:
                mujoco.mjr_setBuffer(
                    mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context
                )

        if save_frames_to_path and self.frame_rate:
            self.save_frames_to_path = save_frames_to_path
            self._initialize_video_writer()
            #os.makedirs(save_frames_to_path, exist_ok=True)
            self.frame_count = 0
        else:
            self.save_frames_to_path = None

    def _initialize_video_writer(self):
        width, height = glfw.get_framebuffer_size(self.window)
        os.makedirs(os.path.dirname(self.save_frames_to_path), exist_ok=True)
        vfps = 10*self.frame_rate
        if self.render_lossless:
            self.container = av.open(f"{self.save_frames_to_path}.mov", 'w')
            self.stream = self.container.add_stream('rawvideo', rate=vfps)
            self.stream.pix_fmt = "rgb24"
        else:
            self.container = av.open(f"{self.save_frames_to_path}.mp4", 'w')
            self.stream = self.container.add_stream('h264', rate=vfps)
            self.stream.pix_fmt = 'yuv420p'
            self.stream.options = {'crf': '10'}
            #self.container = av.open(f"{self.save_frames_to_path}.mp4", 'w')
            #self.stream = self.container.add_stream('mpeg4', rate=vfps)
            #self.stream.bit_rate = 40000000
        self.stream.width = width
        self.stream.height = height
        # Set up FFmpeg filter graph for video processing
        self.filter_graph = av.filter.Graph()

        # Convert width and height to strings
        width_str = str(width)
        height_str = str(height)

        self.buffer_source = self.filter_graph.add(
            "buffer",
            name="in",
            width=width_str,  # Convert to string
            height=height_str,  # Convert to string
            pix_fmt="rgb24",
            time_base=f"{1/vfps}"
        )
        vflip_filter = self.filter_graph.add("vflip", name="vflip")
        self.buffer_sink = self.filter_graph.add("buffersink", name="out")

        # Link the filters
        self.buffer_source.link_to(vflip_filter)
        vflip_filter.link_to(self.buffer_sink)

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
                self.model, mujoco.mjtFontScale.mjFONTSCALE_100.value
        )
        viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window
        )
        self.viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    def _render(self, figures=None, geoms=None):
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

            if geoms:
                for geom in geoms:
                    geom.draw(self.scene)

            mujoco.mjr_render(self.viewport, self.scene, self.context)

            if figures:
                for fig in figures:
                    mujoco.mjr_figure(fig.viewport, fig.figure, self.context)
                    if fig.ylabel:
                        mujoco.mjr_overlay(
                            mujoco.mjtFont.mjFONT_NORMAL,
                            mujoco.mjtGridPos.mjGRID_LEFT,
                            fig.viewport,
                            fig.ylabel,
                            None,
                            self.context
                        )

           # Add timestamp text
            timestamp = f"Eps: {int(self.eps)}, Time: {self.data.time:.2f}s"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOM,
                mujoco.mjr_maxViewport(self.context),
                timestamp,
                None,
                self.context
            )

            if self.save_frames_to_path:
                width, height = glfw.get_framebuffer_size(self.window)
                image = np.zeros((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(image, None, self.viewport, self.context)
                frame = av.VideoFrame.from_ndarray(
                    image, format='rgb24'
                )
                self.buffer_source.push(frame)
                flipped = self.buffer_sink.pull()
                packet = self.stream.encode(flipped)
                if packet:
                    self.container.mux(packet)

                #image = cv2.cvtColor(np.flip(image, axis=0), cv2.COLOR_RGB2BGR)
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
        if self.save_frames_to_path:
            self.container.close()
