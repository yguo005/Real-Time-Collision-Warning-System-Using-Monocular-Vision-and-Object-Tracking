# CS5330 Final Project
# YUNYU Guo
# April 23 2025

# Monocular Time-to-Collision Estimation System
#
# This code estimates Time-to-Collision (TTC) using a single camera by:
# - Detecting objects using YOLOv5
# - Tracking objects across frames with BoT-SORT
# - Estimating depth using Monodepth2
# - Computing TTC based on depth changes over time
#
# Warning system uses a 3-frame filter to prevent flickering between
# green (safe) and red (warning) bounding boxes. The display shows
# both object detections with TTC values and a color-coded depth map.

''' For Colab:
# Install monodepth2 and dependencies
!git clone https://github.com/nianticlabs/monodepth2.git
%cd monodepth2
# Install required Python libraries
!pip install torch torchvision numpy opencv-python pillow matplotlib boxmot
# use the mono_1024x320 model (model needs to upload to colab)
%cd /content
!mkdir -p models/mono_1024x320
!mv encoder.pth models/mono_1024x320/
!mv depth.pth models/mono_1024x320/
'''
# Load YOLOv5, BoXMOT, and monodepth2
import os
import sys
import time
# add monodepth2 to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "monodepth2")) 
import torch
import cv2
import numpy as np
from pathlib import Path
from boxmot import BotSort
import collections
from networks import ResnetEncoder, DepthDecoder # network s module is part of monodepth2 repository, git clone https://github.com/nianticlabs/monodepth2.git
import PIL.Image as pil
# use a pre-downloaded YOLOv5 model:git clone https://github.com/ultralytics/yolov5.git


# Load YOLOv5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
yolo = torch.hub.load('./yolov5', 'yolov5s', source='local').to(device)
yolo.eval()

# Load BoXMOT
tracker = BotSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device=device,
    half=False
)

# Load monodepth2
# colab: encoder_path = "models/mono_1024x320/encoder.pth"
encoder_path = "mono_1024x320/encoder.pth"
# colab: depth_decoder_path = "models/mono_1024x320/depth.pth"
depth_decoder_path = "mono_1024x320/depth.pth"
encoder = ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()

# Helper for Depth Estimation
def estimate_depth(frame):
    # Resize and preprocess
    original_height, original_width = frame.shape[:2]
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (feed_width, feed_height))
    input_image = pil.fromarray(input_image)
    input_tensor = torch.from_numpy(np.array(input_image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        depth = disp_resized.squeeze().cpu().numpy()
    return depth  # Relative depth map

# Compute Time to Collision (TTC)
def compute_ttc(distances, times):
    if len(distances) < 2:
        return None
    v = (distances[-2] - distances[-1]) / (times[-1] - times[-2])
    if v > 0:
        return distances[-1] / v
    else:
        return None

# Main function to process video
def run_ttc_on_video(video_path, fps=30):
    vid = cv2.VideoCapture(video_path)
    object_depth_history = collections.defaultdict(list)  # id: list of (time, depth)
    critical_frames = collections.defaultdict(int)  # id: count of critical frames
    frame_idx = 0

    # Track timing statistics
    timing_stats = {
        "detection": [],
        "tracking": [],
        "depth": [],
        "ttc": []
    }

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # 1. Detection
        t_start = time.time()
        results = yolo(frame)
        preds = results.xyxy[0].cpu().numpy()
        dets = preds[:, :6] if len(preds) > 0 else np.empty((0, 6))
        timing_stats["detection"].append(time.time() - t_start)

        # 2. Tracking
        t_start = time.time()
        tracks = tracker.update(dets, frame)
        timing_stats["tracking"].append(time.time() - t_start)

        # 3. Depth
        t_start = time.time()
        depth_map = estimate_depth(frame)
        timing_stats["depth"].append(time.time() - t_start)

        # Normalize depth map for visualization
        normalized_depth_map = (depth_map / depth_map.max() * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_JET)

        # 4. TTC calculation
        t_start = time.time()
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Compute median depth for the object
            obj_depth = np.median(depth_map[y1:y2, x1:x2])
            object_depth_history[track_id].append((frame_idx / fps, obj_depth))

            # Compute TTC if possible
            times, depths = zip(*object_depth_history[track_id])
            ttc = compute_ttc(list(depths), list(times))

            # Track how many consecutive frames the TTC is below the threshold
            if track_id not in critical_frames:
                critical_frames[track_id] = 0

            if ttc is not None and ttc < 2.0:
                critical_frames[track_id] += 1
            else:
                critical_frames[track_id] = 0

            # Change color only if TTC is below the threshold for 3 consecutive frames
            color = (0, 255, 0) if critical_frames[track_id] < 3 else (0, 0, 255)

            # Draw bounding box and TTC on the frame
            label = f"ID: {track_id}, TTC: {ttc:.2f}s" if ttc is not None else f"ID: {track_id}, TTC: N/A"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        timing_stats["ttc"].append(time.time() - t_start)

        # 5. Combine video frame and depth map side by side
        combined_display = cv2.hconcat([frame, depth_colormap])  # Combine video and depth map horizontally

        # 6. Display the combined output
        cv2.imshow("Video and Depth Map", combined_display)

        # Wait for a short period to simulate video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 1 ms delay between frames
            break

        frame_idx += 1

    # print timing statistics
    print("\n--- Component Processing Times ---")
    for component, times in timing_stats.items():
        if times:
            avg_time = sum(times) / len(times) * 1000  # convert to ms
            print(f"{component.capitalize()}: {avg_time:.1f} ms/frame")
    print(f"Total: {sum(sum(times) for times in timing_stats.values()) / len(timing_stats['detection']) * 1000:.1f} ms/frame")

    vid.release()
    cv2.destroyAllWindows()  


if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(__file__), "video5.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        print(f"Running TTC estimation on video: {video_path}")
        run_ttc_on_video(video_path)
