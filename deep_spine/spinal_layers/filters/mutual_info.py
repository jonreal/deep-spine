import numpy as np
from collections import deque

class MutualInformationFilter:
    def __init__(self, window_size=1000, num_bins=10):
        self.window_size = window_size
        self.num_bins = num_bins
        self.x_window = deque(maxlen=window_size)
        self.y_window = deque(maxlen=window_size)
        self.joint_hist = np.zeros((num_bins, num_bins))
        self.x_hist = np.zeros(num_bins)
        self.y_hist = np.zeros(num_bins)

    def filter(self, x, y):
        # Add new data points
        self.x_window.append(x)
        self.y_window.append(y)

        # Update histograms
        x_bin = min(int(x * self.num_bins), self.num_bins - 1)
        y_bin = min(int(y * self.num_bins), self.num_bins - 1)

        self.joint_hist[x_bin, y_bin] += 1
        self.x_hist[x_bin] += 1
        self.y_hist[y_bin] += 1

        # If window is full, remove oldest point
        if len(self.x_window) == self.window_size:
            old_x = self.x_window[0]
            old_y = self.y_window[0]
            old_x_bin = min(int(old_x * self.num_bins), self.num_bins - 1)
            old_y_bin = min(int(old_y * self.num_bins), self.num_bins - 1)

            self.joint_hist[old_x_bin, old_y_bin] -= 1
            self.x_hist[old_x_bin] -= 1
            self.y_hist[old_y_bin] -= 1

        return self.compute_mi()

    def compute_mi(self):
        # Compute probabilities
        p_joint = self.joint_hist / np.sum(self.joint_hist)
        p_x = self.x_hist / np.sum(self.x_hist)
        p_y = self.y_hist / np.sum(self.y_hist)
        # Compute mutual information
        mi = 0
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                if p_joint[i, j] > 0:
                    mi += (
                        p_joint[i, j]
                        * np.log2(p_joint[i, j] / (p_x[i] * p_y[j]))
                    )
        return mi
