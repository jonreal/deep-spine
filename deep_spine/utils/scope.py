import socket
import re
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.gridspec as gridspec
import cv2

import socket
import fcntl, os


class VerticalStackPlot:
    def __init__(
        self,
        figsize = (5,8),
        buffer_size = 100,
        trace_list = None,
        data_str = None,
        start_index = None,
        end_index = None,
        ylim = None
    ):

        plt.rcParams['lines.linewidth'] = 0.5
        self.fig = plt.figure(figsize=figsize)

        num_plots = len(trace_list)
        outer_grid = gridspec.GridSpec(
            num_plots, 1, height_ratios=[1] * num_plots
        )

        dt = 1/200
        t = np.arange(0, buffer_size * dt, dt)

        for i, traces in enumerate(trace_list):
            ax = plt.Subplot(self.fig, outer_grid[i])
            self.fig.add_subplot(ax)
            d = np.zeros((buffer_size, traces))
            ax.plot(t,d)
            ax.axhline(y = 0, color = 'k', linestyle = '-')
            if end_index:
                plt.xlim(win[0],win[1])
            if ylim:
                if len(ylim) != 1:
                    plt.ylim(ylim[i][0],ylim[i][1])
                else:
                    plt.ylim(ylim[0][0],ylim[0][1])
            if data_str:
                plt.ylabel(data_str[i])
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_xlabel('')

        axes = plt.gcf().axes
        parent = axes[0]
        for i in range(1,len(axes)):
            axes[i].sharex(parent)
        for ax in axes[:-1]:
            ax.tick_params(bottom=False, labelbottom=False)

        plt.xlabel('time (s)')
        num_digits = len(str(int(t[-1])))
        scale = 10 ** (num_digits - 1)
        ax.set_xticks(np.arange(0, t[-1] + scale, scale))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        self.fig.tight_layout()
        #self.fig.canvas.flush_events()

        self.start_time = time.time()
        self.last_time = self.start_time
        self.frame_count = 0

    def update(self, data):
        now = time.time()
        for i, d in enumerate(data):
            ax = self.fig.axes[i]
            num_traces = d.shape[1]

            max_in_buffer = -np.inf
            min_in_buffer = np.inf

            for j in range(num_traces):
                line = ax.lines[j]
                old_ydata = line.get_ydata()

                # Efficient buffer update
                new_ydata = np.roll(old_ydata, -d.shape[0])
                new_ydata[-d.shape[0]:] = d[:, j]
                line.set_ydata(new_ydata)

                # Efficient min/max calculations
                max_in_buffer = max(max_in_buffer, new_ydata.max())
                min_in_buffer = min(min_in_buffer, new_ydata.min())

            ax.set_ylim(min(min_in_buffer * 1.1, 0), max_in_buffer * 1.1)

        self.fig.canvas.draw_idle()
        #self.fig.canvas.flush_events()

        if now - self.last_time > 10:
            print(
                    f"Rate: {self.frame_count/(now - self.last_time):.2f} fps, "
                    f"Elapsed time: {now - self.start_time:.2f} s"
            )
            self.last_time = now
            self.frame_count = 0
        self.frame_count += 1

def get_sorted_files(prefix):
    # Extract directory and prefix
    directory, base_prefix = os.path.split(prefix)
    directory = directory or "."  # Default to current directory if not provided

    # Regular expression to match files with the given prefix and a numeric suffix
    pattern = re.compile(rf"{re.escape(base_prefix)}(\d+)$")

    # List all files in the specified directory
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Directory {directory} not found.")
        return []

    # Filter and sort files based on the numeric suffix
    matching_files = sorted(
        [os.path.join(directory, f) for f in files if pattern.match(f)],
        key=lambda x: int(pattern.search(os.path.basename(x)).group(1))
    )

    return matching_files

class ScopeBroadcaster:
    def __init__(
        self,
        port=None,
    ):
        if port:
            self.port = port
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def broadcast(self, data):
        databytes = data.tobytes()
        self.sock.sendto(databytes, ('', self.port))

class Scope:
    def __init__(
        self,
        port=None,
        host=None,
        filename=None,
        plot_trial=False,
        save_frames_to_path=None,
        prefix=None,
        buffer_size=100,
        config=None
    ):
        self.filename = filename
        self.save_frames_to_path = save_frames_to_path
        self.prefix = prefix
        self.host = host
        self.port = port
        self.plot_trial = plot_trial

        if self.filename:
            self.file = open(filename, 'r')
            self.file.seek(0, 2)  # Move the pointer to the end of the file
            self.last_size = os.path.getsize(filename)  # Store the initial file size
            self.last_pos = self.file.tell()

        if self.host and self.port:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((host, port))
            self.sock.setblocking(False)  # Non-blocking mode

        self.buffer_size = buffer_size

        if config:
            with open(config, 'r') as f:
                self.config = json.load(f)
        self.datasize = 0
        trace_list = []
        for key, value in self.config.items():
            self.datasize += value
            trace_list.append(value)

        self.data = np.empty((0,self.datasize))  # Assuming 132 columns in data

        # Set up the plot
        plt.ion()  # Enable interactive mode for real-time plotting
        plt.rcParams['lines.linewidth'] = 0.5
        #self.fig = plt.figure(figsize=(5,8))

        self.plot = VerticalStackPlot(
            trace_list=trace_list,
            data_str=list(self.config.keys()),
            buffer_size=self.buffer_size,
        )

        #self.plot = VerticalStackPlot(
        #    num_plots=len(self.config),
        #    buffer_size=self.buffer_size,
        #    dt=0.005,
        #    ylim=[(-1, 1)] * len(self.config),
        #    labels=list(self.config.keys())
        #)

        if self.save_frames_to_path:
            self._initialize_video_writer()
            self.data_files = get_sorted_files(self.prefix)

        self.run(interval=0.1)

    def detect_file_change(self):
        """Detect if the file has been modified (e.g., truncated)."""
        current_size = os.path.getsize(self.filename)
        if current_size != self.last_size:
            self.last_size = current_size
            return True
        return False

    def read_udp_data(self):
        data = []
        try:
            while True:
                try:
                    databytes, addr = self.sock.recvfrom(4096)
                    if databytes:
                        data.append(databytes)
                    else:
                        break
                except BlockingIOError:
                    break
        except ValueError as e:
            print(f"Error parsing data: {e}")

    def read_udp_data(self):
        data = []
        try:
            while True:
                try:
                    databytes, addr = self.sock.recvfrom(4096)
                    if databytes:
                        # Convert bytes to numpy array of floats
                        packet_data = np.frombuffer(databytes, dtype=np.float64)
                        if packet_data.shape == (self.datasize,):
                            data.append(packet_data)
                        else:
                            print(f"Unexpected packet shape: {packet_data.shape}")
                    else:
                        break
                except BlockingIOError:
                    break
        except ValueError as e:
            print(f"Error parsing data: {e}")
        # Convert the list of packets to a 2D numpy array
        if data:
            return np.stack(data, axis=0)  # Stack packets along a new axis
        return np.empty((self.datasize, 0))  # Return an empty array if no data
        # Concatenate all collected data bytes

    def read_new_data(self):
        if self.detect_file_change():
            self.last_pos = self.file.tell()
            new_lines = self.file.readlines()
            new_data = []
            for line in new_lines:
                try:
                    row = np.array([float(x) for x in line.strip().split(',')])
                    if row.size == self.datasize:
                        new_data.append(row)
                except ValueError as e:
                    continue
            self.last_pos = self.file.tell()
            return np.array(new_data)
        return np.empty((0, self.datasize))

    def update_data(self, new_data):
        if new_data.size == 0:
            return
        # Ensure new_data is a 2D array
        if new_data.ndim == 1:
            new_data = new_data[np.newaxis, :]  # Convert (N,) to (1, N)
        # Check if new_data has the correct number of columns
        if new_data.shape[1] != self.datasize:
            raise ValueError(
                f"New data has incorrect number of columns:"
                f" {new_data.shape[1]} != {self.datasize}"
            )
        # Append new_data to self.data
        self.data = np.vstack((self.data, new_data))
        # Trim the buffer to keep only the latest `buffer_size` rows
        if self.data.shape[0] > self.buffer_size:
            self.data = self.data[-self.buffer_size:]

    def data_parser(self, data):
        try:
            data_arrays = []
            data_strs = []
            n = 0;
            for key, size in self.config.items():
                data_arrays.append(data[:,n:n+size])
                data_strs.append(key)
                n += size
            return data_arrays, data_strs
        except IndexError as e:
            print(f"Error parsing data: {e}")
            return [], []

    def redraw_plot(self):
        """
        Clears and redraws the plot with the updated data.
        """
        self.plot.fig.clf()
        data, data_str = self.data_parser(self.data)
        plot_vertical_stack(self.plot.fig, data=data, data_str=data_str)
        plt.draw()
        plt.pause(0.1)

    def _initialize_video_writer(self):
        os.makedirs(os.path.dirname(self.save_frames_to_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = int(self.plot.fig.bbox.width), int(self.plot.fig.bbox.height)
        self.video_writer = cv2.VideoWriter(
            f"{self.save_frames_to_path}.mp4",
            fourcc, 100.0, (width, height)
        )


    def run(self, interval=0.5):
        plt.ion()  # Enable interactive mode
        if self.plot_trial:
            self.plot_whole_file()
            return
        elif self.save_frames_to_path:
            self.make_video_from_data_files()
            return
        try:
            while True:
                if self.filename:
                    new_data = self.read_new_data()
                elif self.host and self.port:
                    new_data = self.read_udp_data()
                if new_data.size > 0:
                    self.update_data(new_data)
                    self.redraw_plot()
                time.sleep(interval)  # Adjust polling interval if needed
        except KeyboardInterrupt:
            # Keep the plot window open after interruption
            plt.ioff()  # Disable interactive mode
            plt.show()  # Show the plot window

    def plot_whole_file(self):
        print("Plotting the whole file...")
        self.file.seek(0)
        data = np.loadtxt(self.file, delimiter=',')
        data, data_str = self.data_parser(data)
        plot_vertical_stack(data=data, data_str=data_str)
        plt.draw()
        plt.pause(1)
        plt.ioff()  # Disable interactive mode
        plt.show()  # Show the plot window


    def make_video_from_data_files(self, interval_factor=10):
        if not self.data_files:
            print("No data files found matching the prefix.")
            return

        current_time_step = 0
        time_step_duration = 1 / 200
        interval_duration = time_step_duration * interval_factor

        print(
            f"Starting video creation with interval:"
            f"{interval_duration} seconds per frame."
        )

        for file_path in self.data_files:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        # Parse the line into a numpy array
                        row = np.array(
                            [float(x) for x in line.strip().split(',')]
                        )
                        if row.size == self.datasize:
                            # Update data buffer
                            self.update_data(row)

                            # Determine if this time step should produce a frame
                            if (
                                (current_time_step != 0)
                                and (current_time_step % interval_factor == 0)
                            ):
                                # Redraw plot
                                data, data_str = self.data_parser(self.data)
                                self.plot.update(data)

                                #self.redraw_plot()
                                # Capture the frame from the plot
                                self.plot.fig.canvas.draw()
                                frame = np.frombuffer(
                                    self.plot.fig.canvas.tostring_rgb(),
                                    dtype=np.uint8
                                )
                                frame = frame.reshape(
                                    self.plot.fig.canvas.get_width_height()[::-1]
                                    + (3,)
                                )
                                # Write the frame to the video
                                self.video_writer.write(cv2.cvtColor(
                                    frame, cv2.COLOR_RGB2BGR)
                                )

                            # Increment the time step counter
                            current_time_step += 1

                        else:
                            print(f"Skipping row with unexpected size: {row.size}")

                    except ValueError as e:
                        print(f"Error parsing line: {e}")

        # Release the video writer
        self.video_writer.release()
        print("Video creation complete.")

def parse_args():
    """
    Parses command-line arguments for filename and buffer size.
    """
    parser = argparse.ArgumentParser(description="Real-Time Data Plotter")
    parser.add_argument('-f', '--filename',
        type=str, required=False, help="Path to the data file."
    )
    parser.add_argument('-n', '--buffersize',
        type=int, default=100, help="Buffer size for plotting (default: 100)."
    )
    parser.add_argument('-c', '--config',
        type=str, default=None, help="Path to the config file."
    )
    parser.add_argument('-t', '--plot_trial',
        action='store_true', default=False, help="Flag to enable plotting the trial."
    )
    parser.add_argument('-u', '--host',
        type=str, default=None, help="Host IP address."
    )
    parser.add_argument('-p', '--port',
        type=int, default=None, help="Port number."
    )
    parser.add_argument('-s', '--save_frames_to_path',
        type=str, default=None, help="Flag to enable video."
    )

    # Add a required 'prefix' argument, but only if `--save_frames_to_path` is used
    parser.add_argument(
        '--prefix', 
        type=str, 
        default=None, 
        help="Prefix name of the data files. Required for --save_frames_to_path.")

    # Parse arguments
    args = parser.parse_args()

    # Validate dependency
    if args.save_frames_to_path and not args.prefix:
        parser.error("--save_frames_to_path requires --prefix to be specified.")

    return args



# Example usage
if __name__ == '__main__':
    args = parse_args()
    scope = Scope(
        filename=args.filename,
        save_frames_to_path=args.save_frames_to_path,
        prefix=args.prefix,
        host=args.host,
        port=args.port,
        plot_trial=args.plot_trial,
        buffer_size=args.buffersize,
        config=args.config
    )
