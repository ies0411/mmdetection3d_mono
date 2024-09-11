auto_scale_lr = dict(base_batch_size=32, enable=False)
backbone_norm_cfg = dict(requires_grad=True, type='LN')
backend_args = None
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
custom_imports = dict(imports=[
    'projects.MonoDETR.monodetr',
])
data_prefix = dict(img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')
data_root = '/mnt/nas3/Data/nuScenes/v1.0-trainval/'
dataset_type = 'NuScenesDataset'
db_sampler = dict(
    backend_args=None,
    classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
    data_root='/mnt/nas3/Data/nuScenes/v1.0-trainval/',
    info_path=
    '/mnt/nas3/Data/nuScenes/v1.0-trainval/nuscenes_dbinfos_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(
            barrier=5,
            bicycle=5,
            bus=5,
            car=5,
            construction_vehicle=5,
            motorcycle=5,
            pedestrian=5,
            traffic_cone=5,
            trailer=5,
            truck=5)),
    rate=1.0,
    sample_groups=dict(
        barrier=2,
        bicycle=6,
        bus=4,
        car=2,
        construction_vehicle=7,
        motorcycle=6,
        pedestrian=2,
        traffic_cone=2,
        trailer=6,
        truck=3))
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        sweeps_num=10,
        test_mode=True,
        type='LoadPointsFromMultiSweeps'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
find_unused_parameters = False
ida_aug_conf = dict(
    H=900,
    W=1600,
    bot_pct_lim=(
        0.0,
        0.0,
    ),
    final_dim=(
        320,
        800,
    ),
    rand_flip=True,
    resize_lim=(
        0.47,
        0.625,
    ),
    rot_lim=(
        0.0,
        0.0,
    ))
img_norm_cfg = dict(
    mean=[
        103.53,
        116.28,
        123.675,
    ],
    std=[
        57.375,
        57.12,
        58.395,
    ],
    to_rgb=False)
input_modality = dict(use_camera=True, use_lidar=True)
launcher = 'none'
load_from = '/mnt/d/fcos3d_vovnet_imgbackbone-remapped.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0001
metainfo = dict(classes=[
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
])
model = dict(
    backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        ),
        style='pytorch',
        type='Backbone'),
    type='MonoDETR')
num_epochs = 24
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(T_max=24, by_epoch=True, type='CosineAnnealingLR'),
]
point_cloud_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=1)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            img='',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                to_float32=True,
                type='LoadMultiViewImageFromFiles'),
            dict(
                data_aug_conf=dict(
                    H=900,
                    W=1600,
                    bot_pct_lim=(
                        0.0,
                        0.0,
                    ),
                    final_dim=(
                        320,
                        800,
                    ),
                    rand_flip=True,
                    resize_lim=(
                        0.47,
                        0.625,
                    ),
                    rot_lim=(
                        0.0,
                        0.0,
                    )),
                training=False,
                type='ResizeCropFlipImage'),
            dict(keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset',
        use_valid_flag=True),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(
        backend_args=None, to_float32=True,
        type='LoadMultiViewImageFromFiles'),
    dict(
        data_aug_conf=dict(
            H=900,
            W=1600,
            bot_pct_lim=(
                0.0,
                0.0,
            ),
            final_dim=(
                320,
                800,
            ),
            rand_flip=True,
            resize_lim=(
                0.47,
                0.625,
            ),
            rot_lim=(
                0.0,
                0.0,
            )),
        training=False,
        type='ResizeCropFlipImage'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=24)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            img='',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                to_float32=True,
                type='LoadMultiViewImageFromFiles'),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'construction_vehicle',
                    'bus',
                    'trailer',
                    'barrier',
                    'motorcycle',
                    'bicycle',
                    'pedestrian',
                    'traffic_cone',
                ],
                type='ObjectNameFilter'),
            dict(
                data_aug_conf=dict(
                    H=900,
                    W=1600,
                    bot_pct_lim=(
                        0.0,
                        0.0,
                    ),
                    final_dim=(
                        320,
                        800,
                    ),
                    rand_flip=True,
                    resize_lim=(
                        0.47,
                        0.625,
                    ),
                    rot_lim=(
                        0.0,
                        0.0,
                    )),
                training=True,
                type='ResizeCropFlipImage'),
            dict(
                reverse_angle=False,
                rot_range=[
                    -0.3925,
                    0.3925,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                training=True,
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTransImage'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_bboxes_labels',
                    'attr_labels',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'centers_2d',
                    'depths',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset',
        use_valid_flag=True),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None, to_float32=True,
        type='LoadMultiViewImageFromFiles'),
    dict(
        type='LoadAnnotations3D',
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ],
        type='ObjectNameFilter'),
    dict(
        data_aug_conf=dict(
            H=900,
            W=1600,
            bot_pct_lim=(
                0.0,
                0.0,
            ),
            final_dim=(
                320,
                800,
            ),
            rand_flip=True,
            resize_lim=(
                0.47,
                0.625,
            ),
            rot_lim=(
                0.0,
                0.0,
            )),
        training=True,
        type='ResizeCropFlipImage'),
    dict(
        reverse_angle=False,
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        training=True,
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTransImage'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'attr_labels',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'centers_2d',
            'depths',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            img='',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                to_float32=True,
                type='LoadMultiViewImageFromFiles'),
            dict(
                data_aug_conf=dict(
                    H=900,
                    W=1600,
                    bot_pct_lim=(
                        0.0,
                        0.0,
                    ),
                    final_dim=(
                        320,
                        800,
                    ),
                    rand_flip=True,
                    resize_lim=(
                        0.47,
                        0.625,
                    ),
                    rot_lim=(
                        0.0,
                        0.0,
                    )),
                training=False,
                type='ResizeCropFlipImage'),
            dict(keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset',
        use_valid_flag=True),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.2,
    0.2,
    8,
]
work_dir = 'work_dir/test'
