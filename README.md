Real Time Collision Warning System Using Monocular Vision and Object Tracking 

This project estimates the Time-to-Collision (TTC) with objects using only a single camera feed.
See presentation video: https://northeastern-my.sharepoint.com/:v:/g/personal/guo_yunyu_northeastern_edu/EcWEl74IVOdCkNjEBaDCcuUBgxIOC83npvu9aFdVGAfb_w?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=dzj84T

Overview

Detects objects using YOLOv5.
Tracks objects across frames using BoT-SORT.
Estimates depth using Monodepth2.
Calculates TTC based on changes in estimated depth.
Visualizes results with bounding boxes (color-coded for TTC warning) and a depth map.
Includes an evaluation script (TTC_evaluation.py) to test accuracy against the KITTI dataset.

How it Works

Detection: YOLOv5 finds objects in each frame.
Tracking: BoT-SORT assigns IDs to track objects over time.
Depth Estimation: Monodepth2 creates a depth map for the scene.
TTC Calculation:
Extracts median depth for each tracked object.
Calculates velocity based on depth change.
Computes TTC = depth / velocity.
Uses a 3-frame filter to stabilize warnings (red box for TTC < 2.0s).

Running the Demo

Setup:

Clone yolov5 and monodepth2 repositories into the project folder.
Install required libraries: torch, opencv-python, numpy, matplotlib, pillow, boxmot.
Download pretrained models (osnet_x0_25_msmt17.pt, encoder.pth, depth.pth) and place them in the correct directories (., mono_1024x320).

Run:

python time_to_collisions.py
Press 'q' to quit the visualization.

Evaluation

Download the KITTI Depth Completion dataset (https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip) and extract it into the depth_selection folder.
Run the evaluation script:
This will print depth and TTC accuracy metrics and save error histograms (ttc_error_histograms.png).
