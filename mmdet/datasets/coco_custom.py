# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Union
import copy

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

print("Registering COCOCustomDataset")

@DATASETS.register_module()
class COCOCustomDataset(BaseDetDataset):
	"""Custom COCO Dataset for MMSegmentation."""

	METAINFO = {
		'classes': ['NeuriteSoma', 'OutOfBound', 'Soma'],
		'palette': [
		[0, 0, 0],        # background - black
		[128, 0, 0],      # NeuriteSoma - dark red
		[0, 128, 0],      # OutOfBound - dark green
		[128, 128, 0]     # Soma - olive
	]
	}
	COCOAPI = COCO
	ANN_ID_UNIQUE = True  # Add this property
	# def __init__(self, img_prefix, seg_prefix=None, **kwargs):
	# 	# Prepare the data_prefix dictionary
	# 	data_prefix = {
	# 		'img_path': img_prefix,
	# 		'seg_map_path': seg_prefix if seg_prefix else ''
	# 	}

	# 	# Call the parent class __init__
	# 	super().__init__(
	# 		img_suffix='.png',  # Ensure correct image format
	# 		seg_map_suffix='.png',  # Ensure correct mask format
	# 		reduce_zero_label=False,  # Keep class 0 as background
	# 		data_prefix=data_prefix,  # Pass img and seg paths properly
	# 		**kwargs  # Pass any other keyword arguments
	# 	)
	def load_data_list(self) -> List[dict]:
		"""Load annotations from an annotation file."""
		with get_local_path(
				self.ann_file, backend_args=self.backend_args) as local_path:
			self.coco = self.COCOAPI(local_path)
			
		# Get category IDs that match your class names
		self.cat_ids = self.coco.get_cat_ids(
			cat_names=self.metainfo['classes'])  # Skip background class
		self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
		self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

		img_ids = self.coco.get_img_ids()
		data_list = []
		total_ann_ids = []
		
		for img_id in img_ids:
			raw_img_info = self.coco.load_imgs([img_id])[0]
			raw_img_info['img_id'] = img_id

			ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
			raw_ann_info = self.coco.load_anns(ann_ids)
			total_ann_ids.extend(ann_ids)

			parsed_data_info = self.parse_data_info({
				'raw_ann_info': raw_ann_info,
				'raw_img_info': raw_img_info
			})
			data_list.append(parsed_data_info)
			
		if self.ANN_ID_UNIQUE:
			assert len(set(total_ann_ids)) == len(
				total_ann_ids
			), f"Annotation ids in '{self.ann_file}' are not unique!"

		del self.coco
		return data_list

	def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
		"""Parse raw annotations to the standard format."""
		img_info = raw_data_info['raw_img_info']
		ann_info = raw_data_info['raw_ann_info']

		data_info = {}

		# Set paths
		img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
		if self.data_prefix.get('seg', None):
			seg_map_path = osp.join(
				self.data_prefix['seg'],
				img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
		else:
			seg_map_path = None
		
		data_info['img_path'] = img_path
		data_info['img_id'] = img_info['img_id']
		data_info['seg_map_path'] = seg_map_path
		data_info['height'] = img_info['height']
		data_info['width'] = img_info['width']

		instances = []
		for i, ann in enumerate(ann_info):
			instance = {}

			if ann.get('ignore', False):
				continue
				
			# Handle bounding box
			x1, y1, w, h = ann['bbox']
			inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
			inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
			if inter_w * inter_h == 0:
				continue
			if ann['area'] <= 0 or w < 1 or h < 1:
				continue
			if ann['category_id'] not in self.cat_ids:
				continue
				
			bbox = [x1, y1, x1 + w, y1 + h]

			if ann.get('iscrowd', False):
				instance['ignore_flag'] = 1
			else:
				instance['ignore_flag'] = 0
				
			instance['bbox'] = bbox
			instance['bbox_label'] = self.cat2label[ann['category_id']]

			# Handle segmentation mask
			if ann.get('segmentation', None):
				instance['mask'] = ann['segmentation']

			instances.append(instance)
			
		data_info['instances'] = instances
		return data_info
