import mmcv
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization.local_visualizer import DetLocalVisualizer
import torch
import re

# Path to config file and checkpoint file
config_file = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former\20250603_123001\vis_data\config.py"
checkpoint_file = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former\best_coco_segm_mAP_50_iter_550000.pth"
filename = os.path.basename(checkpoint_file)

# Search for iter number
match = re.search(r"iter_\d+", filename)
if match:
    iteration = match.group()
# Initialize the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# Path to the input image
img_path = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\FIV906_neurons\quartered\t1_B05_s4_w1_z1_bottom_left.bmp"
img_path2 = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\Kao_Allison\Vacor-1a_exp\other_ex_images_1a - bmps\crops_512\4h_A - 3(fld 1 wv 405 - Orange)_bl.bmp"

# Get parent directory and output directory
parent_dir = os.path.dirname(img_path)
parent_dir2 = os.path.dirname(img_path2)
save_dir = os.path.dirname(config_file)
output_dir = os.path.join(save_dir, fr"results_cp_{iteration}")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the parent directory
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
image_files = [f for f in os.listdir(parent_dir) if os.path.splitext(f)[1].lower() in image_extensions]
image_files2 = [f for f in os.listdir(parent_dir2) if os.path.splitext(f)[1].lower() in image_extensions]

# Process each image
for image_file in image_files:
    print("image_file ", image_file)
    img_full_path = os.path.join(parent_dir, image_file)
    result = inference_detector(model, img_full_path)
    img_data = mmcv.imread(img_full_path)

    # Initialize LocalVisualizer
    visualizer = DetLocalVisualizer(alpha=0.3, line_width=1)
    
    # Define output path
    out_file = os.path.join(output_dir, f"result_{image_file}")

    # Save image with detections
    visualizer.add_datasample(
        name='result',
        draw_pred=True,
        pred_score_thr=0.4,
        image=img_data,
        data_sample=result,
        draw_gt=True,   # Don't draw ground truth, just predictions
        show=False,      # Don't show the image interactively
        out_file=out_file  # Save the processed image
    )

    print(f"Processed and saved: {out_file}")
    
for image_file in image_files2:
    print("image_file ", image_file)
    img_full_path = os.path.join(parent_dir2, image_file)
    result = inference_detector(model, img_full_path)
    img_data = mmcv.imread(img_full_path)

    # Initialize LocalVisualizer
    visualizer = DetLocalVisualizer(alpha=0.3, line_width=1)
    
    # Define output path
    out_file = os.path.join(output_dir, f"result_{image_file}")

    # Save image with detections
    visualizer.add_datasample(
        name='result',
        draw_pred=True,
        pred_score_thr=0.4,
        image=img_data,
        data_sample=result,
        draw_gt=True,   # Don't draw ground truth, just predictions
        show=False,      # Don't show the image interactively
        out_file=out_file  # Save the processed image
    )

    print(f"Processed and saved: {out_file}")
print("Processing complete!")