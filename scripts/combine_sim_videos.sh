#!/bin/bash

# Input files
sim="animations/sim.mp4"
weights="animations/weights.mp4"
traces="animations/sim-traces.mp4"

# Output files
weights_scaled="weights_scaled.mp4"
traces_scaled="traces_scaled.mp4"
traces_resized="traces_resized.mp4"
final_output="animations/combined.mp4"

# Scale weights to the desired size (e.g., to inset size: 300x200)
ffmpeg -i "$weights" -vf "scale=450:300" -c:v libx264 -crf 23 -preset fast "$weights_scaled"

# Adjust weights playback speed to match the length of sim
ffmpeg -i "$sim" -i "$weights_scaled" -filter_complex \
"[0:v]setpts=PTS-STARTPTS[sim]; \
 [1:v]setpts=(15/5)*PTS[weights]; \
 [sim][weights]overlay=10:10" \
-c:v libx264 -crf 23 -preset fast weights_adjusted.mp4

# Adjust traces playback speed to match the length of sim
ffmpeg -i "$traces" -filter:v "setpts=(30/5)*PTS" -c:v libx264 -crf 23 -preset fast "$traces_scaled"

# Resize traces to match the height of sim
ffmpeg -i "$traces_scaled" -vf "scale=-2:900" -c:v libx264 -crf 23 -preset fast "$traces_resized"

# Combine sim with traces and overlay weights as an inset
ffmpeg -i "$sim" -i "$weights_scaled" -i "$traces_resized" -filter_complex \
"[0:v][2:v]hstack=inputs=2[main]; \
 [main][1:v]overlay=10:10" \
-c:v libx264 -crf 23 -preset fast "$final_output"

# Cleanup intermediate files
rm -f "$weights_scaled" "$traces_scaled" "$traces_resized"

#!/bin/bash

# Input files
sim="animations/sim.mp4"
weights="animations/weights.mp4"
traces="animations/sim-traces.mp4"

# Output files
weights_scaled="weights_scaled.mp4"
traces_resized="traces_resized.mp4"
final_output="animations/combined.mp4"

# Scale and adjust playback speed for weights (larger size: 450x300)
ffmpeg -i "$weights" -filter_complex \
"scale=450:300:force_original_aspect_ratio=decrease,pad=450:300:(ow-iw)/2:(oh-ih)/2,setpts=(15/5)*PTS" \
-c:v libx264 -crf 23 -preset fast "$weights_scaled"

# Scale and adjust playback speed for traces (combine scaling and timing)
ffmpeg -i "$traces" -filter_complex \
"scale=-2:900,setpts=(30/5)*PTS" \
-c:v libx264 -crf 23 -preset fast "$traces_resized"

# Combine sim with traces and overlay weights as an inset
ffmpeg -i "$sim" -i "$weights_scaled" -i "$traces_resized" -filter_complex \
"[0:v][2:v]hstack=inputs=2[main]; \
 [main][1:v]overlay=10:10" \
-c:v libx264 -crf 23 -preset fast "$final_output"

# Cleanup intermediate files
rm -f "$weights_scaled" "$traces_resized"
