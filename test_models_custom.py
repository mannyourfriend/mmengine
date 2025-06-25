import mmcv
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization.local_visualizer import DetLocalVisualizer
import torch
import re
import cv2
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from torchvision.ops import masks_to_boxes
from datetime import datetime
import json

# Path to config file and checkpoint file
config_file = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former\20250603_123001\vis_data\config.py"
checkpoint_file = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former\20250603_123001\vis_data\best_coco_segm_mAP_50_iter_550000.pth"
ROTATION_ANGLES_COUNT = 20   # feel free to trim
MASK_SCORE_THR  = 0.01
FUSE_IOU_THR    = 0.15
MIN_SUPPORT     = 12
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
output_dir = os.path.join(save_dir, fr"results_cp_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

categories = {0: "NeuriteSoma", 1: "OutOfBound", 2: "Soma", 3: "Cluster"}
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the parent directory
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
image_files = [f for f in os.listdir(parent_dir) if os.path.splitext(f)[1].lower() in image_extensions]
image_files2 = [f for f in os.listdir(parent_dir2) if os.path.splitext(f)[1].lower() in image_extensions]
def rotation_angles(num_angles: int):
    """
    Return a list of evenly spaced rotation angles that cover 360°.

    Examples
    --------
    >>> rotation_angles(4)
    [0, 90, 180, 270]

    >>> rotation_angles(6)
    [0, 60, 120, 180, 240, 300]
    """
    if num_angles < 1:
        raise ValueError("num_angles must be ≥ 1")

    step = 360.0 / num_angles
    # round() makes 0, 90, 180, … exactly ints for common divisors of 360
    return [round(i * step, 6) for i in range(num_angles)]

def sample_from_fused(fused_polys, img_h, img_w):
    """
    fused_polys: list[(Polygon, score, label)]
    Returns a DetDataSample ready for DetLocalVisualizer.
    """
    if not fused_polys:
        return DetDataSample()        # empty sample → visualizer shows raw img

    # 1. build dense masks  -------------------------------
    masks = []
    for poly, _, _, _ in fused_polys:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array(list(poly.exterior.coords)).astype(int)
        cv2.fillPoly(mask, [pts], 1)          # fill = 1
        masks.append(mask)
    masks = np.stack(masks)                   # (N, H, W) uint8

    # 2. pack into InstanceData ---------------------------
    inst = InstanceData()
    inst.masks  = torch.from_numpy(masks) > 0        # bool tensor
    inst.scores = torch.tensor([s for _, s, _, _ in fused_polys])
    inst.labels = torch.tensor([l for _, _, l, _ in fused_polys])

    # optional bboxes (visualizer uses them for score text placement)
    inst.bboxes = masks_to_boxes(inst.masks.float())

    # 3. wrap in DetDataSample ----------------------------
    return DetDataSample(pred_instances=inst)

def bitmap_to_polygon(bitmap):
	"""Convert masks from the form of bitmaps to polygons.

	Args:
		bitmap (ndarray): masks in bitmap representation.

	Return:
		list[ndarray]: the converted mask in polygon representation.
		bool: whether the mask has holes.
	"""
	bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
	# cv2.RETR_CCOMP: retrieves all of the contours and organizes them
	#   into a two-level hierarchy. At the top level, there are external
	#   boundaries of the components. At the second level, there are
	#   boundaries of the holes. If there is another contour inside a hole
	#   of a connected component, it is still put at the top level.
	# cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
	outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	contours = outs[-2]
	hierarchy = outs[-1]
	if hierarchy is None:
		return [], False
	# hierarchy[i]: 4 elements, for the indexes of next, previous,
	# parent, or nested contours. If there is no corresponding contour,
	# it will be -1.
	with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
	contours = [c.reshape(-1, 2) for c in contours]
	return contours, with_hole

# Process each image
def rotate_image(img: np.ndarray, angle: float):
	"""Return rotated image + forward & inverse affine matrices."""
	h, w = img.shape[:2]
	center = (w / 2, h / 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	invM = cv2.invertAffineTransform(M)
	rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
	return rot, M, invM

def derotate_mask(mask: np.ndarray, invM, out_shape):
	"""Warp a binary mask back to the original orientation."""
	return cv2.warpAffine(mask.astype(np.uint8), invM, out_shape,
						flags=cv2.INTER_NEAREST).astype(bool)

class FilledMaskVisualizer(DetLocalVisualizer):
    """Draw masks as filled, semi-transparent blobs instead of just edges."""

    def _draw_instances(self,
                        image,
                        instances,
                        classes=None,
                        palette=None,
                        **kwargs):
        # Call parent to draw boxes / labels first (they'll go on top later)
        img = super()._draw_instances(
            image, instances, classes, palette, show_mask=False, **kwargs)

        if not hasattr(instances, 'masks') or instances.masks.numel() == 0:
            return img

        masks  = instances.masks.cpu().numpy()
        colors = self._get_palette(palette, len(masks))

        for i, mask in enumerate(masks):
            contours, _ = bitmap_to_polygon(mask)
            # <── key difference: pass face_colors
            img = self.draw_polygons(
                contours,
                edge_colors=colors[i],      # outline
                face_colors=colors[i],      # fill
                alpha=0.4,                  # 0 = transparent, 1 = opaque
                line_width=self.line_width,
                canvas=img)

        return img

# ---------- polygon helpers --------------------------------------------------
def polygons_from_mask(mask_bool: np.ndarray):
	"""Uses MMDet's stock bitmap_to_polygon (already in the import path)."""
	contours, _ = bitmap_to_polygon(mask_bool)
	polys = []
	for c in contours:
		if len(c) < 3:
			continue
		p = Polygon(c)

		# Fix self-intersections etc.  Two standard tricks:
		if not p.is_valid:
			# ① classic buffer(0) trick
			p = p.buffer(0)

			# ② for Shapely ≥2, make_valid is available:
			# from shapely.make_valid import make_valid
			# p = make_valid(p)

		# Some geometries may split into MultiPolygon after buffer(0);
		# flatten them and keep only areas > 0
		if p.is_empty:
			continue
		if p.geom_type == 'Polygon':
			polys.append(p)
		else:  # MultiPolygon or GeometryCollection
			polys.extend([geom for geom in p.geoms if geom.area > 0])

	return [Polygon(c) for c in contours if len(c) >= 3]

def iou(p1: Polygon, p2: Polygon):
	if not (p1.is_valid and p2.is_valid):
		return 0.0
	inter = p1.intersection(p2).area
	return inter / p1.union(p2).area if inter > 0 else 0.0

coco_gt = COCO()           # empty constructor
coco_gt.dataset = coco_dict
coco_gt.createIndex()

model_classes   = list(model.dataset_meta['classes'])   # e.g. ['axon', 'dendrite']
coco_name2id    = {cat['name']: cat['id']
                   for cat in coco_gt.loadCats(coco_gt.getCatIds())}

model2coco = {i: coco_name2id.get(name)
              for i, name in enumerate(model_classes)}

for image_file in image_files:
	img_full_path = os.path.join(parent_dir, image_file)
	img_orig      = mmcv.imread(img_full_path)
	h, w          = img_orig.shape[:2]

	# ── 1) run model on each rotation & collect derotated polygons ─────────
	collected = []                                     # [(Polygon, score, label)]
	for ang in rotation_angles(ROTATION_ANGLES_COUNT):
		rot_img, M, invM = rotate_image(img_orig, ang)
		det_sample = inference_detector(model, rot_img).cpu()

		if not hasattr(det_sample.pred_instances, 'masks'):
			continue                                   # model has no masks

		inst = det_sample.pred_instances
		keep = inst.scores > MASK_SCORE_THR
		inst = inst[keep]

		if inst.masks.numel() == 0:
			continue                                   # nothing above thr

		for m, s, l in zip(inst.masks.bool(),
						inst.scores,
						inst.labels):
			mask_back = derotate_mask(m.numpy(), invM, (w, h))
			for poly in polygons_from_mask(mask_back):
				collected.append((ang, poly, float(s), int(l)))

	# ── 2) Fuse polygons across rotations by IoU ----------------------------
	fused = []   # [(Polygon, best_score, label, support_set)]
	for rot_id, poly, scr, lab in collected:
		merged = False
		for i, (fp, fs, fl, support) in enumerate(fused):
			if lab != fl:
				continue
			if iou(poly, fp) >= FUSE_IOU_THR:
				new_poly   = fp.union(poly)
				new_score  = max(fs, scr)
				new_supp   = support | {rot_id}          # union of sets
				fused[i]   = (new_poly, new_score, lab, new_supp)
				merged = True
				break
		if not merged:
			fused.append((poly, scr, lab, {rot_id}))
	fused = [t for t in fused if len(t[3]) >= MIN_SUPPORT]

	# ── 3) Build an MMDet sample & pretty overlay ---------------------------
	fused_sample = sample_from_fused(fused, h, w)
	visualizer = DetLocalVisualizer(alpha=0.3, line_width=0.3)
	visualizer.dataset_meta = model.dataset_meta
	# visualizer = FilledMaskVisualizer(
	# 	alpha=0.3,                     # transparency for bboxes
	# 	line_width=1,
	# 	# dataset_meta=model.dataset_meta  # keeps class colours consistent
	# )
	visualizer.add_datasample(
		name='rotFuse',               # window name (ignored because show=False)
		image=img_orig,               # original image
		data_sample=fused_sample,     # our fused predictions
		draw_pred=True,               # draw them
		pred_score_thr=0.0,           # already filtered → show all
		draw_gt=True,
		show=False,
		out_file=os.path.join(output_dir, f"rotFuse_{image_file}")
	)
	print("Saved overlay →", os.path.join(output_dir, f"rotFuse_{image_file}"))

	
for image_file in image_files2:
	img_full_path = os.path.join(parent_dir2, image_file)
	img_orig      = mmcv.imread(img_full_path)
	h, w          = img_orig.shape[:2]

	# ── 1) run model on each rotation & collect derotated polygons ─────────
	collected = []                                     # [(Polygon, score, label)]
	for ang in rotation_angles(ROTATION_ANGLES_COUNT):
		rot_img, M, invM = rotate_image(img_orig, ang)
		det_sample = inference_detector(model, rot_img).cpu()

		if not hasattr(det_sample.pred_instances, 'masks'):
			continue                                   # model has no masks

		inst = det_sample.pred_instances
		keep = inst.scores > MASK_SCORE_THR
		inst = inst[keep]

		if inst.masks.numel() == 0:
			continue                                   # nothing above thr

		for m, s, l in zip(inst.masks.bool(),
						inst.scores,
						inst.labels):
			mask_back = derotate_mask(m.numpy(), invM, (w, h))
			for poly in polygons_from_mask(mask_back):
				collected.append((ang, poly, float(s), int(l)))

	# ── 2) Fuse polygons across rotations by IoU ----------------------------
	fused = []   # [(Polygon, best_score, label, support_set)]
	for rot_id, poly, scr, lab in collected:
		merged = False
		for i, (fp, fs, fl, support) in enumerate(fused):
			if lab != fl:
				continue
			if iou(poly, fp) >= FUSE_IOU_THR:
				new_poly   = fp.union(poly)
				new_score  = (fs+scr) / 2
				new_supp   = support | {rot_id}          # union of sets
				fused[i]   = (new_poly, new_score, lab, new_supp)
				merged = True
				break
		if not merged:
			fused.append((poly, scr, lab, {rot_id}))
	fused = [t for t in fused if len(t[3]) >= MIN_SUPPORT]

	# ── 3) Build an MMDet sample & pretty overlay ---------------------------
	fused_sample = sample_from_fused(fused, h, w)

	visualizer = DetLocalVisualizer(alpha=0.3, line_width=0.3)
	# visualizer = FilledMaskVisualizer(
	# 	alpha=0.3,                     # transparency for bboxes
	# 	line_width=1,
	# 	# dataset_meta=model.dataset_meta  # keeps class colours consistent
	# )	
	visualizer.add_datasample(
		name='rotFuse',               # window name (ignored because show=False)
		image=img_orig,               # original image
		data_sample=fused_sample,     # our fused predictions
		draw_pred=True,               # draw them
		pred_score_thr=0.0,           # already filtered → show all
		draw_gt=True,
		show=False,
		out_file=os.path.join(output_dir, f"rotFuse_{image_file}")
	)
	print("Saved overlay →", os.path.join(output_dir, f"rotFuse_{image_file}"))

run_cfg = dict(
	timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S"),
	config_file    = config_file,
	checkpoint_file= checkpoint_file,
	rotation_count = ROTATION_ANGLES_COUNT,
	rotation_angles= rotation_angles(ROTATION_ANGLES_COUNT),  # ← uses helper
	mask_score_thr = MASK_SCORE_THR,
	fuse_iou_thr   = FUSE_IOU_THR,
	min_support    = MIN_SUPPORT,
)

# ---------------------------------------------------------------
# 2. Choose where to save it; here we keep it next to the results
# ---------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)
meta_path = os.path.join(output_dir, f"run_meta_{run_cfg['timestamp']}.json")

# ---------------------------------------------------------------
# 3. Dump as pretty-printed JSON
# ---------------------------------------------------------------
with open(meta_path, "w", encoding="utf-8") as fh:
	json.dump(run_cfg, fh, indent=4)

print(f"[meta] saved run configuration → {meta_path}")