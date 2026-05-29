import os, json, re, mmcv, torch, numpy as np
import sys
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import socket
from datetime import datetime
import shapely

# your existing helpers ----------------------------
from rotation_utils import rotate_image, derotate_mask, polygons_from_mask
from rotation_utils import rotation_angles     # the helper we wrote earlier
from rotation_utils import iou                  # if you still need it
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.apis import init_detector, inference_detector

# --------------------------------------------------
# Path to config file and checkpoint file
def parse_arguments():
	"""Parse command line arguments for config and checkpoint paths."""
	parser = argparse.ArgumentParser(description='Model Evaluation Script')
	parser.add_argument('--config', type=str, required=True,
					help='Path to the config file (.py)')
	parser.add_argument('--checkpoint', type=str, required=True,
					help='Path to the checkpoint file (.pth)')
	return parser.parse_args()

# Replace your hardcoded paths with this:
if __name__ == "__main__":
	# Check if arguments are provided, otherwise use defaults for manual running
	if len(sys.argv) > 1:
		# Running with command line arguments
		args = parse_arguments()
		CONFIG_FILE = args.config
		CHECKPOINT_FILE = args.checkpoint
	else:
		# Running manually - keep your existing hardcoded paths as fallback
		CONFIG_FILE = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former_10neuron_noSoma_run1\custom_mask2former_10neuron_noSoma.py"
		CHECKPOINT_FILE = r"C:\Users\five\Desktop\Manny\AI_only\mmdetection\work_dirs\custom_mask2former_10neuron_noSoma_run1\best_coco_segm_mAP_50_epoch_20.pth"
	
	print(f"Using config file: {CONFIG_FILE}")
	print(f"Using checkpoint file: {CHECKPOINT_FILE}")
	# JSON_GT         = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\HandAnnotations_inprogress\handAnnos\annotations\instances_default_NoClusters_CrowdSoma.json"   # path to the uploaded file
	JSON_GT = r"S:\Phys\FIV906 NeuroArbors\Real_Neurons\HandAnnotations_inprogress\handAnnos\annotations\instances_default_NoClusters_CrowdSoma.json"
	filename = os.path.basename(CHECKPOINT_FILE)
	save_dir = os.path.join(os.path.dirname(CONFIG_FILE), "test_model_hand_annos")
	match = re.search(r"iter_\d+", filename)
	if not match:
		match = re.search(r"epoch_\d+", filename)
	iteration = match.group()
	timeID = datetime.now().strftime("%Y%m%d%H%M%S")

	output_dir = os.path.join(save_dir, fr"results_cp_{iteration}_ID_{timeID}")
	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	OUT_JSON        = os.path.join(output_dir, "coco_results.json")
	ROTATION_ANGLES_COUNT = 1  # feel free to trim
	MASK_SCORE_THR  = 0.2   # matches your visualizer
	FUSE_IOU_THR    = 0.2
	MIN_SUPPORT     = 1
	# ROTATION_ANGLES_COUNT = 1   # feel free to trim
	# MASK_SCORE_THR  = 0.1  # matches your visualizer
	# FUSE_IOU_THR    = 0.2
	# MIN_SUPPORT     = int(ROTATION_ANGLES_COUNT * 0.4)
	DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

	metadata = {
		"outputFile": OUT_JSON,
		"numRotations": ROTATION_ANGLES_COUNT,
		"maskScoreThreshold": MASK_SCORE_THR,
		"IOU_FusionThreshold": FUSE_IOU_THR,
		"minimumMatchingObjs": MIN_SUPPORT,
		"onGPU": DEVICE=='cuda',
		"computerName": socket.gethostname(),
		"modelPath": CHECKPOINT_FILE,
		"labelPath": JSON_GT,
		"timeID": timeID
	}
	metadataFile = os.path.join(output_dir, f"metadata_ID_{timeID}.json")
	with open(metadataFile, 'w', encoding='utf-8') as fh:
		json.dump(metadata, fh, indent=2)

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

	# missing = [i for i, cid in model2coco.items() if cid is None]
	# if missing:
	# 	raise RuntimeError(f"Model labels {missing} not found in COCO categories.")
	# coco_dt = coco_gt.loadRes(OUT_JSON)

	# print("  ↳ GT  images :", len(coco_gt.getImgIds()))
	# print("  ↳ Pred images:", len(coco_dt.getImgIds()))
	# print("  ↳ GT  cats   :", coco_gt.getCatIds())
	# print("  ↳ Pred cats  :", coco_dt.getCatIds())

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


	for metric in ['bbox', 'segm']:
		evaluator = COCOeval(coco_gt, coco_dt, metric)
		evaluator.evaluate()
		evaluator.accumulate()
		print(f"\n=====  {metric.upper()}  =====")
		evaluator.summarize()