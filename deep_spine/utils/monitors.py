import cv2
from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BaseMonitor:
    def __init__(self, save_frames_to_path):
        self.save_frames_to_path = save_frames_to_path

    def _initialize_video_writer(self):
        os.makedirs(os.path.dirname(self.save_frames_to_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = int(self.fig.bbox.width), int(self.fig.bbox.height)
        video_writer = cv2.VideoWriter(
            f"{self.save_frames_to_path}.mp4",
            fourcc, 1, (width, height)
        )
        return video_writer

    def write_frame(self):
        # Convert the Matplotlib figure to a format OpenCV can write to the video
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.video_writer.write(img)

    @abstractmethod
    def _initialize_plot(self):
        raise NotImplementedError


class WeightMonitor(BaseMonitor):
    def __init__(
        self,
        W,
        mn_labels=None,
        ia_labels=None,
        scale_opt=None,
        title=None,
        **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.W = W.detach().cpu().numpy()
        self.mn_labels = mn_labels
        self.ia_labels = ia_labels
        self.scale_opt = scale_opt
        self.fig, self.ax, self.im = self._initialize_plot(title)
        self.video_writer = self._initialize_video_writer()

    def _initialize_plot(self, title):
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = self._plot_heatmap(fig, ax, title)
        return fig, ax, im

    def _plot_heatmap(self, fig, ax, title=None):
        if ax.collections:
            for cbar in ax.collections:
                if cbar.colorbar:
                    cbar.colorbar.remove()
        ax.clear()
        if self.scale_opt:
            im = sns.heatmap(
                self.W, annot=True, fmt='.2f',
                cmap='coolwarm', center=0,
                vmin=-self.scale_opt, vmax=self.scale_opt,
                yticklabels=self.mn_labels,
                xticklabels=self.ia_labels, ax=ax, cbar=True
            )
        else:
            im = sns.heatmap(
                self.W, annot=True, fmt='.2f',
                cmap='coolwarm', vmin=0, vmax=1,
                yticklabels=self.mn_labels,
                xticklabels=self.ia_labels, ax=ax, cbar=True
            )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title(title, pad=-50)
        plt.draw()
        return im

    def update_plot(self, W, title=None):
        self.W = W.detach().cpu().numpy()
        self.im = self._plot_heatmap(self.fig, self.ax, title)
        self.write_frame()

    def close_video(self):
        self.video_writer.release()

