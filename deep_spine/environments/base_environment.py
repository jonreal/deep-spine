import av
import os
import numpy as np
import mujoco
from mujoco.glfw import glfw
from typing import Optional
from PIL import Image
import OpenGL.GL as gl
import time
import copy

import subprocess
from fractions import Fraction


class BaseEnvironment(object):
    def __init__(
        self,
        xml_path: str = 'arm26.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        render_lossless: bool = False,
        video_scale: float = 0.5,
        video_crf: str = '23',
        window_width: int = 1200,
        window_height: int = 900,
        azimuth: int = 180,
        elevation: int = -90,
        distance: int = 2,
        lookat: np.ndarray = np.array([0.35, 0, 0.25]),
        ):
        local_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_path = os.path.join(local_dir, xml_path)
        self.frame_rate = frame_rate
        self.video_scale = video_scale
        self.video_crf = video_crf
        self._initialize_mujoco()

        self.figs = None
        self.geoms = None
        self.annotations = None

        if self.frame_rate:
            self.render_lossless = render_lossless
            self._initialize_render(
                window_width, window_height, azimuth,
                elevation, distance, lookat
            )
        if save_frames_to_path and self.frame_rate:
            self.save_frames_to_path = save_frames_to_path
            self._initialize_video_writer()
            #os.makedirs(save_frames_to_path, exist_ok=True)
            self.frame_count = 0
        else:
            self.save_frames_to_path = None

    def add_geom(self, geom):
        if self.geoms is None:
            self.geoms = []
        self.geoms.append(geom)

    def add_fig(self, fig):
        if self.figs is None:
            self.figs = []
        self.figs.append(fig)

    def add_annotation(self, annotation):
        if self.annotations is None:
            self.annotations = []
        self.annotations.append(annotation)

    def _initialize_video_writer(self):
        width, height = glfw.get_framebuffer_size(self.window)
        os.makedirs(os.path.dirname(self.save_frames_to_path), exist_ok=True)
        self.vfps = 10*self.frame_rate
        if self.render_lossless:
            self.container = av.open(f"{self.save_frames_to_path}.mov", 'w')
            self.stream = self.container.add_stream('rawvideo', rate=self.vfps)
            self.stream.pix_fmt = "rgb24"
        else:

            print(self.video_crf);
            self.container = av.open(f"{self.save_frames_to_path}.mp4", 'w')
            self.stream = self.container.add_stream('h264', rate=self.vfps)
            self.stream.pix_fmt = 'yuv420p'
            self.stream.options = {
                'crf': str(self.video_crf),
                'preset': 'slow'
            }
            #self.container = av.open(f"{self.save_frames_to_path}.mp4", 'w')
            #self.stream = self.container.add_stream('mpeg4', rate=self.vfps)
            #self.stream.bit_rate = 40000000
        # Calculate target dimensions based on scale factor
        target_width = int(width * self.video_scale)
        target_height = int(height * self.video_scale)

        # Set the stream dimensions to the target size
        self.stream.width = target_width
        self.stream.height = target_height

        # Set up FFmpeg filter graph for video processing
        self.filter_graph = av.filter.Graph()

        # Convert width and height to strings
        width_str = str(width)
        height_str = str(height)
        target_width_str = str(target_width)
        target_height_str = str(target_height)

        self.buffer_source = self.filter_graph.add(
            "buffer",
            name="in",
            width=width_str,
            height=height_str,
            pix_fmt="rgb24",
            time_base=f"{1/self.vfps}"
        )

        # Add filters: vflip and scale
        vflip_filter = self.filter_graph.add("vflip", name="vflip")
        scale_filter = self.filter_graph.add("scale", name="scale",
                                            w=target_width_str,
                                            h=target_height_str)
        self.buffer_sink = self.filter_graph.add("buffersink", name="out")

        # Link the filters in sequence
        self.buffer_source.link_to(vflip_filter)
        vflip_filter.link_to(scale_filter)
        scale_filter.link_to(self.buffer_sink)

    def _initialize_mujoco(self):
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def _initialize_render(
        self, window_width, window_height, azimuth, elevation, distance, lookat
        ):
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.distance = distance
        self.cam.lookat = lookat
        glfw.init()
        print(window_width, window_height)
        self.window = glfw.create_window(
            window_width,
            window_height,
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

            if self.geoms is not None:
                for geom in self.geoms:
                    geom.draw(self.scene)

            mujoco.mjr_render(self.viewport, self.scene, self.context)

            if self.figs is not None:
                for fig in self.figs:
                    mujoco.mjr_figure(fig.viewport, fig.figure, self.context)
                    label_viewport = copy.deepcopy(fig.viewport)
                    label_viewport.left += -20

                    if fig.ylabel:
                        x_pixel = label_viewport.left
                        y_pixel = (
                            label_viewport.bottom
                        )
                        #mujoco.mjr_text(
                        #    mujoco.mjtFont.mjFONT_NORMAL,
                        #    fig.ylabel,
                        #    self.context,
                        #    0,
                        #    0,
                        #    1, 1, 1
                        #)
                        mujoco.mjr_overlay(
                            mujoco.mjtFont.mjFONT_NORMAL,
                            mujoco.mjtGridPos.mjGRID_LEFT,
                            label_viewport,
                            fig.ylabel,
                            None,
                            self.context
                        )

            if self.annotations is not None:
                for annotation in self.annotations:
                    mujoco.mjr_overlay(
                        annotation.font,
                        annotation.gridpos,
                        annotation.viewport,
                        annotation.text,
                        None,
                        self.context
                    )

           # Add timestamp text
            timestamp = f"Eps: {int(self.eps)}, Time: {self.data.time:.2f}s"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOP,
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
                frame.pts = self.frame_count
                frame.time_base = Fraction(1, self.vfps)
                self.buffer_source.push(frame)
                flipped = self.buffer_sink.pull()
                packet = self.stream.encode(flipped)
                if packet:
                    self.container.mux(packet)
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
            try:
                while True:
                    packet = self.stream.encode(None)
                    if not packet:
                        break
                    self.container.mux(packet)
            except av.error.EOFError:
                # Encoder has already finished; no further action needed
                pass
            finally:
                self.container.close()
