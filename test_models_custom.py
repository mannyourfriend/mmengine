import mmcv
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization.local_visualizer import DetLocalVisualizer
import torch
import re
import cv2
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, LineString
from shapely import make_valid
from collections import defaultdict
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from torchvision.ops import masks_to_boxes
from datetime import datetime
import json

# Path to config file and checkpoint file
CONFIG_FILE = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former_20neuron\custom_mask2former_20neuron.py"
CHECKPOINT_FILE = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former_20neuron\best_coco_segm_mAP_50_epoch_18.pth"


ROTATION_ANGLES_COUNT = 4  
MASK_SCORE_THR  = 0.1   # matches your visualizer
FUSE_IOU_THR_OUT    = 0.1
FUSE_IOU_THR_IN = 0.000004
MIN_SUPPORT     = 0
filename = os.path.basename(CHECKPOINT_FILE)

# Search for iter number
match = re.search(r"iter_\d+", filename)
if not match:
	match = re.search(r"epoch_\d+", filename)
iteration = match.group()
# Initialize the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)

# Path to the input image
# img_path = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\FIV906_neurons\rotate_img\rot72deg.bmp"
# img_path = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\FIV906_neurons\rotate_img\temp\t1_B05_s4_w1_z1_top_right.bmp"
# img_path2 = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\Kao_Allison\Vacor-1a_exp\other_ex_images_1a - bmps\crops_512\4h_A - 3(fld 1 wv 405 - Orange)_bl.bmp"
img_path = r"C:\Users\five\Desktop\Manny\40xImgs\temp\t1_J06_s12_w1_z1 - Copy.bmp"
# Get parent directory and output directory
parent_dir = os.path.dirname(img_path)
# parent_dir2 = os.path.dirname(img_path2)
save_dir = os.path.dirname(os.path.dirname(img_path))
output_dir = os.path.join(save_dir, fr"results_cp_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

categories = {0: "NeuriteSoma", 1: "OutOfBound", 2: "Soma", 3: "Cluster"}
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the parent directory
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
image_files = [f for f in os.listdir(parent_dir) if os.path.splitext(f)[1].lower() in image_extensions]
# image_files2 = [f for f in os.listdir(parent_dir2) if os.path.splitext(f)[1].lower() in image_extensions]
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
	# return [0,72]
	# return [72]
	return [round(i * step, 6) for i in range(num_angles)]

def _iter_polygons(geom):
	"""Yield Polygon parts from Polygon/MultiPolygon/GeometryCollection."""
	if geom.is_empty:
		return
	gtype = geom.geom_type
	if gtype == 'Polygon':
		yield geom
	elif gtype == 'MultiPolygon':
		for g in geom.geoms:
			if not g.is_empty:
				yield g
	elif gtype == 'GeometryCollection':
		for g in geom.geoms:
			# Recurse only into polygonal pieces
			if g.geom_type in ('Polygon', 'MultiPolygon', 'GeometryCollection'):
				yield from _iter_polygons(g)
	# ignore non-polygonal (LineString/Point) fragments silently

def _mask_from_geom(geom, H, W):
	"""Rasterize a (possibly multi-part) geometry into a single HxW uint8 mask.
	Fills exteriors with 1 and punches interiors (holes) back to 0.
	"""
	mask = np.zeros((H, W), dtype=np.uint8)
	for poly in _iter_polygons(geom):
		# robustify polygon
		if not poly.is_valid:
			poly = poly.buffer(0)
			if poly.is_empty:
				continue
		# exterior
		ext = np.array(poly.exterior.coords).round().astype(int)
		if ext.shape[0] >= 3:
			cv2.fillPoly(mask, [ext], 1)
		# holes
		for ring in poly.interiors:
			hole = np.array(ring.coords).round().astype(int)
			if hole.shape[0] >= 3:
				cv2.fillPoly(mask, [hole], 0)
	return mask

def _iter_boundary_lines(geom, include_interior=False):
	"""Yield boundary LineStrings from Polygon/MultiPolygon/GeometryCollection."""
	if geom.is_empty:
		return
	gt = geom.geom_type
	if gt == 'Polygon':
		yield geom.exterior
		if include_interior:
			for ring in geom.interiors:
				yield LineString(ring.coords)
	elif gt == 'MultiPolygon':
		for g in geom.geoms:
			yield from _iter_boundary_lines(g, include_interior)
	elif gt == 'GeometryCollection':
		for g in geom.geoms:
			if g.geom_type in ('Polygon', 'MultiPolygon', 'GeometryCollection'):
				yield from _iter_boundary_lines(g, include_interior)
	# silently ignore non-polygonal fragments (LineString/Point/etc.)

def perimeter_pct_within_edge_band(p, x, width, height, include_interior=False, inside_only=True):
	"""
	Percentage of polygon perimeter within distance x of the image border.

	Parameters
	----------
	p : shapely geometry
		Polygon/MultiPolygon/GeometryCollection in image pixel coords.
	x : float
		Distance (pixels) from the field-of-view edge.
	width, height : int
		Image width and height (pixels).
	include_interior : bool
		If True, include hole perimeters in both total and near-edge length.
		If False (default), use exterior perimeter only.
	inside_only : bool
		If True (default), measure only the band *inside* the image.
		If False, allow the band to extend outside the image as well.

	Returns
	-------
	pct : float
		Percentage in [0, 100].
	near_len, total_len : floats
		Length near edge and total perimeter (same units as pixels).
	"""
	if x <= 0 or p is None or p.is_empty:
		return 0.0, 0.0, 0.0

	fov = box(0, 0, width, height)

	# Build the "edge band" region
	if inside_only:
		# band of thickness ~x inside the image boundary
		# (boundary.buffer(x) creates a 2-sided strip; intersect with fov keeps inside)
		edge_band = fov.boundary.buffer(x).intersection(fov)
	else:
		# include both inside and just-outside the image border
		edge_band = fov.boundary.buffer(x)

	# Sum perimeter lengths
	total_len = 0.0
	near_len = 0.0
	for line in _iter_boundary_lines(p, include_interior=include_interior):
		if line.is_empty:
			continue
		L = line.length
		if L <= 0:
			continue
		total_len += L
		seg = line.intersection(edge_band)
		if not seg.is_empty:
			near_len += seg.length

	if total_len == 0.0:
		return 0.0, 0.0, 0.0

	pct = 100.0 * (near_len / total_len)
	return pct, near_len, total_len

def sample_from_fused(fused_polys, img_h, img_w, size_filter=0):
	"""
	fused_polys: list of tuples like (geometry, score, label) or (geometry, score, label, support)
	size_filter: minimum mask area in pixels; masks smaller than this are dropped.
	Returns a DetDataSample ready for DetLocalVisualizer.
	"""
	masks = []
	scores = []
	labels = []

	for tup in fused_polys:
		# support-aware or not
		if len(tup) >= 4:
			geom, score, label, _, _ = tup
		else:
			geom, score, label = tup

		if geom is None or geom.is_empty:
			continue

		# Build one mask that covers all polygonal parts (handles Multi/Collection)
		m = _mask_from_geom(geom, img_h, img_w)
		area = int(m.sum())
		if area == 0 or area < size_filter:
			continue

		masks.append(m)
		scores.append(float(score))
		labels.append(int(label))

	if not masks:
		return DetDataSample()  # empty: visualizer will show raw image

	masks_np = np.stack(masks, axis=0)                 # (N,H,W) uint8
	inst = InstanceData()
	inst.masks  = torch.from_numpy(masks_np) > 0       # bool tensor
	inst.scores = torch.tensor(scores, dtype=torch.float32)
	inst.labels = torch.tensor(labels, dtype=torch.int64)

	# optional bboxes for nicer label placement
	try:
		inst.bboxes = masks_to_boxes(inst.masks.float())
	except Exception:
		pass

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
	if not p1.is_valid:
		p1 = p1.buffer(0)
		if p1.is_empty: return 0.0
	if not p2.is_valid:
		p2 = p2.buffer(0)
		if p2.is_empty: return 0.0


	print("polys valid")
	inter = p1.intersection(p2).area
	if inter <= 0:
		return 0.0
	union = p1.union(p2).area
	print("inter: ", inter)
	print("union: ", union)
	return inter / union if inter > 0 else 0.0

for image_file in image_files:
	img_full_path = os.path.join(parent_dir, image_file)
	img_orig      = mmcv.imread(img_full_path)
	h, w          = img_orig.shape[:2]
	img_area = h*w
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
			new_polys = polygons_from_mask(mask_back)
			for poly in new_polys:
				if poly.area < (0.001 * img_area):
					continue
				minDistFromEdge = 10
				edgeRemovalThreshold = 50
				pct, near_len, total_len = perimeter_pct_within_edge_band(
					poly, x=minDistFromEdge, width=w, height=h, include_interior=False, inside_only=True
						)
				if pct > edgeRemovalThreshold:
					print(f"{pct:.1f}% of perimeter lies within {minDistFromEdge} px of the image edge.")
					continue
				merged = False
				for i, (p, score, label, a, overlaps) in enumerate(collected):
					if iou(poly, p) >= FUSE_IOU_THR_IN:
						# Merge with existing polygon
						try: 
							merged_poly = poly.union(p)
						except:
							print("merge error.")
						overlaps+=1
						collected[i] = (
							merged_poly,            # Updated polygon
							max(score, float(s)),   # Max score
							label,                  # Same label
							a,                      # Keep the original rotation angle
							overlaps				# Keep track of overlapping objs
						)
						merged = True
						break
				if not merged:
					collected.append((poly, float(s), int(l), ang, 0))

	# ── 2) Fuse polygons across rotations by IoU ----------------------------
	fused = []   # [(Polygon, best_score, label, support_set)]
	for poly, scr, lab, rot_id, overlaps in collected:
		merged = False
		for i, (fp, fs, fl, support, _) in enumerate(fused):
			iouscore = iou(poly,fp)
			print("ious were: ", iouscore)
			if iou(poly, fp) >= FUSE_IOU_THR_OUT:
				print("fusing")
				new_poly   = fp.union(poly)
				new_score  = max(fs, scr)
				new_supp   = support | {rot_id}          # union of sets
				fused[i]   = (new_poly, new_score, lab, new_supp, iouscore)
				merged = True
				break
		if not merged:
			fused.append((poly, scr, lab, {rot_id}, 0.))

	fused = [t for t in fused if len(t[3]) >= MIN_SUPPORT]
	print("en collected", len(collected))
	print("len fused", len(fused))
	print(len([t for t in fused if t[4] > 0.]))


	# ── 3) Build an MMDet sample & pretty overlay ---------------------------
	fused_sample = sample_from_fused(fused, h, w, size_filter = 100)
	visualizer = DetLocalVisualizer(alpha=1.0, line_width=0.3)
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
		# ── 3) Build an MMDet sample & pretty overlay ---------------------------
	collected_sample = sample_from_fused(collected, h, w, size_filter = 100)
	visualizer = DetLocalVisualizer(alpha=1, line_width=0.3)
	visualizer.dataset_meta = model.dataset_meta
	# visualizer = FilledMaskVisualizer(
	# 	alpha=0.3,                     # transparency for bboxes
	# 	line_width=1,
	# 	# dataset_meta=model.dataset_meta  # keeps class colours consistent
	# )
	visualizer.add_datasample(
		name='rotFuse',               # window name (ignored because show=False)
		image=img_orig,               # original image
		data_sample=collected_sample,     # our fused predictions
		draw_pred=True,               # draw them
		pred_score_thr=0.0,           # already filtered → show all
		draw_gt=True,
		show=False,
		out_file=os.path.join(output_dir, f"rotFuse_coll_{image_file}")
	)
	print("Saved overlay →", os.path.join(output_dir, f"rotFuse_coll_{image_file}"))

run_cfg = dict(
	timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S"),
	config_file    = CONFIG_FILE,
	checkpoint_file= CHECKPOINT_FILE,
	rotation_count = ROTATION_ANGLES_COUNT,
	rotation_angles= rotation_angles(ROTATION_ANGLES_COUNT),  # ← uses helper
	mask_score_thr = MASK_SCORE_THR,
	fuse_iou_thr_within_img   = FUSE_IOU_THR_IN,
	fuse_iou_thr_between_imgs   = FUSE_IOU_THR_OUT,
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