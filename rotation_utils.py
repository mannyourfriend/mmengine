from mmdet.visualization.local_visualizer import DetLocalVisualizer
import torch
import cv2
import numpy as np
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from torchvision.ops import masks_to_boxes
from shapely.geometry import Polygon

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

def iou(p1: Polygon, p2: Polygon):
	if not (p1.is_valid and p2.is_valid):
		return 0.0
	inter = p1.intersection(p2).area
	return inter / p1.union(p2).area if inter > 0 else 0.0
        
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