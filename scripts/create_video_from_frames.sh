#!/bin/bash

# Check if the correct number of arguments were passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 path_to_frames prefix output_file"
    exit 1
fi

# Assign command line arguments to variables
path_to_frames="$1"
prefix="$2"
output_file="$3"

# Run ffmpeg to stitch the frames together into a video
ffmpeg -framerate 24 -i "${path_to_frames}/${prefix}%09d.png" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "${output_file}"
