# CS5330 Final Project
# YUNYU Guo
# April 23 2025

# KITTI Depth Evaluation (Local Version)
# Evaluates Monodepth2 predictions and YOLOv5 detections on KITTI depth completion dataset.

import os
import torch
import cv2
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from boxmot import BotSort
from monodepth2.networks import ResnetEncoder, DepthDecoder
import PIL.Image as pil

# Set up paths
project_dir = os.path.dirname(__file__)
kitti_raw_path = os.path.join(project_dir, "depth_selection", "val_selection_cropped", "image")
kitti_gt_path = os.path.join(project_dir, "depth_selection", "val_selection_cropped", "groundtruth_depth")
reid_model_path = Path(project_dir) / "osnet_x0_25_msmt17.pt"
encoder_path = os.path.join(project_dir, "mono_1024x320", "encoder.pth")
depth_decoder_path = os.path.join(project_dir, "mono_1024x320", "depth.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5
yolo = torch.hub.load(os.path.join(project_dir, 'yolov5'), 'yolov5s', source='local').to(device)
yolo.eval()

# Load BoT-SORT
tracker = BotSort(
    reid_weights=reid_model_path,
    device=device,
    half=False
)

# Load Monodepth2
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

print("Models loaded.")

# Helper for Depth Estimation
def estimate_depth(frame):
    original_height, original_width = frame.shape[:2]
    input_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image_resized = cv2.resize(input_image_rgb, (feed_width, feed_height), interpolation=cv2.INTER_LINEAR)
    input_pil = pil.fromarray(input_image_resized)
    input_np = np.array(input_pil).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        pred_disp = disp_resized.squeeze().cpu().numpy()
    pred_depth = 1.0 / np.maximum(pred_disp, 1e-6)
    return pred_depth

# Load KITTI Ground Truth Depth
def load_kitti_gt_depth_completion(filepath):
    depth_png = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        return None
    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = np.nan
    return depth

# Compute depth errors with median scaling
def compute_depth_errors(gt, pred):
    mask = ~np.isnan(gt) & (gt > 0)
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    if len(pred_masked) == 0:
        return None
    scale = np.median(gt_masked) / np.median(pred_masked)
    pred_scaled = pred_masked * scale
    thresh = np.maximum((gt_masked / pred_scaled), (pred_scaled / gt_masked))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(np.mean((gt_masked - pred_scaled) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_masked) - np.log(pred_scaled)) ** 2))
    abs_rel = np.mean(np.abs(gt_masked - pred_scaled) / gt_masked)
    sq_rel = np.mean(((gt_masked - pred_scaled) ** 2) / gt_masked)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# TTC computation
def compute_ttc(depths, times):
    if len(depths) < 2:
        return None
    v = (depths[-2] - depths[-1]) / (times[-1] - times[-2])
    if v <= 0:
        return None
    return depths[-1] / v

# TTC evaluation
def evaluate_ttc(pred_depth_map, gt_depth_map, tracks, frame_idx, fps=10):
    ttc_results = []
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track[:5])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pred_depth_map.shape[1], x2), min(pred_depth_map.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        pred_obj_depth = np.nanmedian(pred_depth_map[y1:y2, x1:x2])
        gt_obj_depth = np.nanmedian(gt_depth_map[y1:y2, x1:x2])
        if np.isnan(pred_obj_depth) or np.isnan(gt_obj_depth):
            continue
        if track_id not in pred_depth_history:
            pred_depth_history[track_id] = []
        if track_id not in gt_depth_history:
            gt_depth_history[track_id] = []
        pred_depth_history[track_id].append((frame_idx / fps, pred_obj_depth))
        gt_depth_history[track_id].append((frame_idx / fps, gt_obj_depth))
        if len(pred_depth_history[track_id]) >= 2:
            pred_times, pred_depths = zip(*pred_depth_history[track_id])
            pred_ttc = compute_ttc(list(pred_depths), list(pred_times))
            gt_times, gt_depths = zip(*gt_depth_history[track_id])
            gt_ttc = compute_ttc(list(gt_depths), list(gt_times))
            if pred_ttc is not None and gt_ttc is not None:
                ttc_results.append({
                    'track_id': track_id,
                    'pred_ttc': pred_ttc,
                    'gt_ttc': gt_ttc,
                    'error': abs(pred_ttc - gt_ttc),
                    'relative_error': abs(pred_ttc - gt_ttc) / gt_ttc if gt_ttc > 0 else float('inf')
                })
    return ttc_results

# Global variables for TTC
pred_depth_history = {}
gt_depth_history = {}

def run_kitti_evaluation(kitti_raw_path, kitti_gt_path, sequence_name):
    print(f"Starting evaluation for sequence: {sequence_name}")
    image_dir = kitti_raw_path
    gt_depth_dir = kitti_gt_path
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    gt_depth_files = sorted(glob.glob(os.path.join(gt_depth_dir, '*.png')))
    print(f"Found {len(image_files)} images and {len(gt_depth_files)} ground truth depth maps.")
    if len(image_files) == 0 or len(gt_depth_files) == 0:
        print("Error: Missing image or ground truth depth files!")
        return
    all_depth_errors = []
    all_ttc_results = []
    for frame_idx, image_path in enumerate(image_files[:50]):  # Limit 50 for demonstration
        print(f"Processing frame {frame_idx + 1}/50: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        gt_depth_path = gt_depth_files[frame_idx]
        gt_depth = load_kitti_gt_depth_completion(gt_depth_path)
        if gt_depth is None:
            continue
        results = yolo(frame)
        preds = results.xyxy[0].cpu().numpy()
        dets = preds[:, :6] if len(preds) > 0 else np.empty((0, 6))
        tracks = tracker.update(dets, frame)
        pred_depth = estimate_depth(frame)
        depth_errors = compute_depth_errors(gt_depth, pred_depth)
        if depth_errors is not None:
            all_depth_errors.append(depth_errors)
        ttc_results = evaluate_ttc(pred_depth, gt_depth, tracks, frame_idx)
        all_ttc_results.extend(ttc_results)
    if all_depth_errors:
        mean_errors = np.array(all_depth_errors).mean(0)
        print("\n--- Depth Evaluation Results ---")
        print(f"Sequence: {sequence_name}")
        print(f"Frames Evaluated: {len(all_depth_errors)}")
        print(("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}").format(
              'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'd1<1.25', 'd2<1.25^2', 'd3<1.25^3'))
        print(("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}").format(
              mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3],
              mean_errors[4], mean_errors[5], mean_errors[6]))
    if all_ttc_results:
        ttc_errors = [r['error'] for r in all_ttc_results]
        ttc_rel_errors = [r['relative_error'] for r in all_ttc_results if r['relative_error'] != float('inf')]
        print("\n--- TTC Evaluation Results ---")
        print(f"Total TTC calculations: {len(all_ttc_results)}")
        print(f"Mean absolute TTC error: {np.mean(ttc_errors):.2f} seconds")
        print(f"Median absolute TTC error: {np.median(ttc_errors):.2f} seconds")
        if ttc_rel_errors:
            print(f"Mean relative TTC error: {np.mean(ttc_rel_errors)*100:.1f}%")
            print(f"Median relative TTC error: {np.median(ttc_rel_errors)*100:.1f}%")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(ttc_errors, bins=20)
        plt.xlabel('Absolute Error (seconds)')
        plt.ylabel('Frequency')
        plt.title('TTC Absolute Error Distribution')
        plt.subplot(1, 2, 2)
        if ttc_rel_errors:
            plt.hist([e*100 for e in ttc_rel_errors], bins=20)
            plt.xlabel('Relative Error (%)')
            plt.ylabel('Frequency')
            plt.title('TTC Relative Error Distribution')
        plt.tight_layout()
        plt.savefig('ttc_error_histograms.png')
        plt.close()  
        plt.show()

def main():
    print("\n====== KITTI Time-to-Collision Evaluation (Local Version) ======\n")
    sequence_name = "val_selection_cropped"
    if not os.path.exists(kitti_raw_path) or not os.path.exists(kitti_gt_path):
        print("ERROR: KITTI depth completion dataset not found.")
        return 1
    print(f"\nStarting evaluation with:")
    print(f"  - Image path: {kitti_raw_path}")
    print(f"  - Ground truth path: {kitti_gt_path}")
    print(f"  - Sequence: {sequence_name}")
    try:
        run_kitti_evaluation(kitti_raw_path, kitti_gt_path, sequence_name)
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
    print("\n====== Evaluation Complete ======\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())