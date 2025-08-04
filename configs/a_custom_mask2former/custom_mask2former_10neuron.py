_base_ = [
	#'../_base_/datasets/cocoCustom.py', 
	'../_base_/default_runtime.py'
]
image_size = (512, 512)
batch_augments = [
	dict(
		type='BatchFixedSizePad',
		size=image_size,
		img_pad_value=0,
		pad_mask=True,
		mask_pad_value=0,
		pad_seg=True,
		seg_pad_value=255)
]

data_preprocessor = dict(
	type='DetDataPreprocessor',
	mean=[47.35, 47.34, 47.34,],
	std=[62.46, 62.46, 62.46],
	bgr_to_rgb=False,
	pad_size_divisor=32,
	pad_mask=True,
	mask_pad_value=0,
	pad_seg=True,
	seg_pad_value=255,
	batch_augments=batch_augments)

num_things_classes = 3
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type='Mask2Former',
    data_preprocessor=data_preprocessor,  # Added explicit size for 512x512 images
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=200,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                # type='DeformableDetrTransformerEncoder',
                num_layers=8,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=8,
                        dropout=0.1,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1536,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=12,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=10),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=15)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12288,  # Adjusted from 18432 for 512x512 images (proportional to area)
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.4,
        filter_low_score=True),
    init_cfg=None)
# dataset settings
backend_args = None
scale = image_size
data_root = r"C:\Users\five\Desktop\Manny\05a11_10neuron/"
dataset_type = 'COCOCustomDataset'

train_pipeline = [
	dict(type='LoadImageFromFile', backend_args=backend_args),
	dict(type='LoadAnnotations', with_bbox=True, with_mask=True, backend_args=backend_args),
	dict(type='Resize', scale=scale, keep_ratio=True),
	dict(type='RandomFlip', prob=0.5),
	dict(type='PackDetInputs')
]
test_pipeline = [
	dict(type='LoadImageFromFile', backend_args=backend_args),
	dict(type='Resize', scale=scale, keep_ratio=True),
	dict(type='LoadAnnotations', backend_args=backend_args),
	dict(
		type='PackDetInputs',
		meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
				'scale_factor'))
]

train_dataloader = dict(
	batch_size=4,
	num_workers=4,
	persistent_workers=False,
	sampler=dict(type='DefaultSampler', shuffle=True),
	batch_sampler=dict(type='AspectRatioBatchSampler'),
	dataset=dict(
		type=dataset_type,
		data_root=data_root,
		ann_file="coco_train.json",
		#img_prefix='png_train',
		#seg_prefix='mask_train_norm/',
		data_prefix=dict(
			img='train/'),#'png_train/'),# seg='train/'),
		filter_cfg=dict(filter_empty_gt=True, min_size=32),
		pipeline=train_pipeline,
		backend_args=backend_args))
val_dataloader = dict(
	batch_size=1,
	num_workers=4,
	persistent_workers=False,
	drop_last=False,
	sampler=dict(type='DefaultSampler', shuffle=False),
	dataset=dict(
		type=dataset_type,
		data_root=data_root,
		ann_file='coco_val.json',
		data_prefix=dict(img='train/'),#'png_train/'),# seg='val/'),img_prefix = 'png_train',#
		# seg_prefix = 'mask_train_norm/',
		test_mode=True,
		pipeline=test_pipeline,
		backend_args=backend_args))
test_dataloader = dict(
	batch_size=4,
	num_workers=4,
	persistent_workers=False,
	drop_last=False,
	sampler=dict(type='DefaultSampler', shuffle=False),
	dataset=dict(
		type=dataset_type,
		data_root=data_root,
		ann_file='coco_test.json',
		data_prefix=dict(img='train/'),#'png_train/'),# seg='test/'),img_prefix = 'png_train',#
		#seg_prefix='mask_train_norm/',
		test_mode=True,
		pipeline=test_pipeline,
		backend_args=backend_args))
monitor_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ValReal_YodaCrunch.json',
        data_prefix=dict(img='valReal/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)



val_evaluator = dict(
	type='CocoMetric',
	ann_file=data_root + 'coco_val.json',
	metric=['bbox', 'segm'],  # Evaluates both bounding boxes and segmentation masks
	backend_args=backend_args)
test_evaluator = dict(
	type='CocoMetric',
	ann_file=data_root + 'coco_test.json',
	metric=['bbox', 'segm'],  # Evaluates both bounding boxes and segmentation masks
	backend_args=backend_args) 	
monitor_evaluator = dict(             # will compute normal COCO numbers
    type='CocoMetric',
    ann_file=data_root + 'ValReal_YodaCrunch.json',
    metric=['bbox', 'segm'],
    backend_args=backend_args)

# val_dataloader  = [val_dataloader,  monitor_dataloader]
# val_evaluator   = [val_evaluator,   monitor_evaluator]

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
	type='OptimWrapper',
	optimizer=dict(
		type='AdamW',
		lr=0.0001,
		weight_decay=0.05,
		eps=1e-8,
		betas=(0.9, 0.999)),
	paramwise_cfg=dict(
		custom_keys={
			'backbone': dict(lr_mult=0.1, decay_mult=1.0),
			'query_embed': embed_multi,
			'query_feat': embed_multi,
			'level_embed': embed_multi,
		},
		norm_decay_mult=0.0),
	clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 3687500
param_scheduler = dict(
	type='MultiStepLR',
	begin=0,
	end=max_iters,
	by_epoch=False,
	milestones=[327778, 355092],
	gamma=0.1)

# Before 365001th iteration, we do evaluation every 20000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 25000 // 4
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
# train_cfg = dict(
# 	type='IterBasedTrainLoop',
# 	max_iters=max_iters,
# 	val_interval=interval,
# 	dynamic_intervals=dynamic_intervals)
train_cfg = dict(
	type='EpochBasedTrainLoop', 
	max_epochs=20, 
	val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
	#logger=dict(type='LoggerHook', interval=50),  # Log every 50 iterations
	checkpoint=dict(
		type='CheckpointHook',
		by_epoch=True,
		save_last=True,
		max_keep_ckpts=20,
		interval=1,
		save_best='coco/segm_mAP_50'),
	#param_scheduler=dict(type='ParamSchedulerHook'),  # For learning rate updates
# timer=dict(type='IterTimerHook'),  # Track iteration time
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)