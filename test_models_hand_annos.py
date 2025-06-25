import os, json, re, mmcv, torch, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from tqdm import tqdm
import cv2

# your existing helpers ----------------------------
from rotation_utils import rotate_image, derotate_mask, polygons_from_mask
from rotation_utils import rotation_angles     # the helper we wrote earlier
from rotation_utils import iou                  # if you still need it
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.apis import init_detector, inference_detector

# --------------------------------------------------
# Path to config file and checkpoint file
CONFIG_FILE = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former\20250603_123001\vis_data\config.py"
CHECKPOINT_FILE = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former\20250603_123001\vis_data\best_coco_segm_mAP_50_iter_550000.pth"
JSON_GT         = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\HandAnnotations_inprogress\handAnnos\annotations\instances_default - Copy.json"    # path to the uploaded file
filename = os.path.basename(CHECKPOINT_FILE)
save_dir = os.path.dirname(CONFIG_FILE)
match = re.search(r"iter_\d+", filename)
if match:
	iteration = match.group()
output_dir = os.path.join(save_dir, fr"results_cp_{iteration}")
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

OUT_JSON        = os.path.join(output_dir, "coco_results.json")
ROTATION_ANGLES_COUNT = 20   # feel free to trim
MASK_SCORE_THR  = 0.01   # matches your visualizer
FUSE_IOU_THR    = 0.15
MIN_SUPPORT     = 12
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'


# Initialize the model
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

coco_gt = COCO(JSON_GT)
img_ids = coco_gt.getImgIds()
model_classes   = list(model.dataset_meta['classes'])   # e.g. ['axon', 'dendrite']
coco_name2id    = {cat['name']: cat['id']
				for cat in coco_gt.loadCats(coco_gt.getCatIds())}
print(coco_name2id)
model2coco = {i: coco_name2id.get(name)
			for i, name in enumerate(model_classes)}

missing = [i for i, cid in model2coco.items() if cid is None]
if missing:
	raise RuntimeError(f"Model labels {missing} not found in COCO categories.")
coco_dt = coco_gt.loadRes(OUT_JSON)

print("  ↳ GT  images :", len(coco_gt.getImgIds()))
print("  ↳ Pred images:", len(coco_dt.getImgIds()))
print("  ↳ GT  cats   :", coco_gt.getCatIds())
print("  ↳ Pred cats  :", coco_dt.getCatIds())

def coco_rle_from_binary(mask_bool):
	"""binary H×W → compressed RLE dict that COCO expects."""
	rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
	rle['counts'] = rle['counts'].decode('ascii')  # bytes→str for JSON
	return rle

def predict_image_coco(img_info):
	img_path = img_info['file_name']          # full path already in JSON
	orig     = mmcv.imread(img_path)
	h, w     = orig.shape[:2]

	# 1) rotate ensemble
	collected = []    # [(rot_id, poly, score, label)]
	for ang in rotation_angles(ROTATION_ANGLES_COUNT):
		rot_img, M, invM = rotate_image(orig, ang)
		sample = inference_detector(model, rot_img).cpu()
		inst   = sample.pred_instances
		keep   = inst.scores > MASK_SCORE_THR
		inst   = inst[keep]

		if hasattr(inst, 'masks') and inst.masks.numel() > 0:
			for m, s, lab in zip(inst.masks.bool(),
								inst.scores,
								inst.labels):
				back = derotate_mask(m.numpy(), invM, (w, h))
				for poly in polygons_from_mask(back):
					collected.append((ang, poly, float(s), int(lab)))

	# 2) fuse
	fused = []   # [(Poly, score, label, support_set)]
	for ang, poly, score, lab in collected:
		merged = False
		for i, (fp, fs, fl, supp) in enumerate(fused):
			if lab!=fl or iou(poly, fp) < FUSE_IOU_THR: continue
			fused[i] = (fp.union(poly), max(fs, score), lab, supp|{ang})
			merged = True
			break
		if not merged:
			fused.append((poly, score, lab, {ang}))

	fused = [t for t in fused if len(t[3]) >= MIN_SUPPORT]

	# 3) convert to COCO result dicts
	results = []
	for poly, score, lab, _ in fused:
		coco_cat = model2coco.get(lab)
		if coco_cat is None:
			continue                     # no matching category in GT

		mask = np.zeros((h, w), dtype=np.uint8)
		cv2.fillPoly(mask, [np.array(list(poly.exterior.coords)).astype(int)], 1)
		rle = coco_rle_from_binary(mask)

		results.append(dict(
			image_id   = img_info['id'],
			category_id= coco_cat,
			score      = score,
			segmentation = rle,             # for segm AP
			bbox         = list(mask_utils.toBbox(rle)),  # for bbox AP
		))
	return results

coco_results = []
for img_id in tqdm(img_ids, desc="Predicting"):
	coco_results.extend(predict_image_coco(coco_gt.loadImgs(img_id)[0]))

# save so you can inspect later
with open(OUT_JSON, "w") as fh:
	json.dump(coco_results, fh)
print(f"[saved] {OUT_JSON}  ({len(coco_results)} detections)")

coco_dt = coco_gt.loadRes(OUT_JSON)

# model_classes   = list(model.dataset_meta['classes'])   # e.g. ['axon', 'dendrite']
# coco_name2id    = {cat['name']: cat['id']
#                    for cat in coco_gt.loadCats(coco_gt.getCatIds())}

# model2coco = {i: coco_name2id.get(name)
#               for i, name in enumerate(model_classes)}

# missing = [i for i, cid in model2coco.items() if cid is None]
# if missing:
#     raise RuntimeError(f"Model labels {missing} not found in COCO categories.")

for metric in ['bbox', 'segm']:
	evaluator = COCOeval(coco_gt, coco_dt, metric)
	evaluator.evaluate()
	evaluator.accumulate()
	print(f"\n=====  {metric.upper()}  =====")
	evaluator.summarize()