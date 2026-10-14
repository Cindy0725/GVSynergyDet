_base_ = [
    '../../../configs/_base_/schedules/mmdet-schedule-1x.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.GVSynergyDet.gvsynergydet.gvsynergydet_arkit',
        'projects.GVSynergyDet.gvsynergydet.indoor_Det_metric',
        'projects.GVSynergyDet.gvsynergydet.indoor_NVS_metric',
        'projects.GVSynergyDet.gvsynergydet.loading',
        'projects.GVSynergyDet.gvsynergydet.processing',
        'projects.GVSynergyDet.gvsynergydet.formating',
        'projects.GVSynergyDet.gvsynergydet.imvoxelheadv2',
        'projects.GVSynergyDet.gvsynergydet.arkit_dataset', 
    ],
    allow_failed_imports=False
)

prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-3.2, -0.2, -2.28, 3.2, 6.2, 0.28]],
    rotations=[.0])


class_names = ["cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool", "chair", "table", "tv_monitor", "sofa"]
classes = len(class_names)

model = dict(
    type='GVSynergyDet_ARKit',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    neck_3d=dict(
        type='IndoorImVoxelNeck',
        in_channels=256,
        out_channels=128,
        n_blocks=[1, 1, 1]),

    bbox_head=dict(
        type='ImVoxelHead_ARKit',
        n_classes=17,
        n_levels=3,
        n_channels=128,
        n_reg_outs=7,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        prior_generator=prior_generator),
    prior_generator=prior_generator,
    n_voxels=[40, 40, 16],
    voxel_size = [0.16, 0.16, 0.2],
    coord_type='DEPTH',
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.25, score_thr=.01))

dataset_type = 'AtlasARKitDataset'
# Root containing the prepared annotations and RGB images.
data_root = 'data/arkit/'

backend_args = None


metainfo = dict(CLASSES=class_names)

train_pipeline = [
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='LoadMultiViewImageFromFilesARKit3DGS', data_root=data_root, load_depth=False, backend_args=backend_args, num_views=56, nerf_target_views=2, loading='gap'),
    dict(type='GVS_Resize', scale=(280, 224), keep_ratio=False),
    dict(type='GVS_Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesARKit3DGS', data_root=data_root, load_depth=False, backend_args=backend_args, num_views=101, nerf_target_views=1, loading='random'),
    dict(type='GVS_Resize', scale=(280, 224), keep_ratio=False),
    dict(type='GVS_Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='arkit_infos_train.pkl',
            modality = dict(use_camera=True, use_lidar=False),
            pipeline=train_pipeline,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Depth',
            metainfo=metainfo,
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='arkit_infos_val.pkl',
        modality = dict(use_camera=True, use_lidar=False),
        pipeline=test_pipeline,
        test_mode=True,
        box_type_3d='Depth',
        metainfo=metainfo,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='GVSynergyDet_IndoorMetric',
    ann_file=data_root + 'arkit_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator



optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]


randomness = dict(seed=2060582775)


# hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', max_keep_ckpts=2, save_best=['mAP_0.25', 'mAP_0.50'], rule="greater"))

# runtime
find_unused_parameters = True 


