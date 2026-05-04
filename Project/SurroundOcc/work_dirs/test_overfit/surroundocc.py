point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', use_semantic=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation'
        ],
        with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation'
        ],
        with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CustomNuScenesOccDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(type='LoadOccupancy', use_semantic=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',
                    'truck', 'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation'
                ],
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
        ],
        classes=[
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        occ_size=[200, 200, 16],
        pc_range=[-50, -50, -5.0, 50, 50, 3.0],
        use_semantic=True),
    val=dict(
        type='CustomNuScenesOccDataset',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadOccupancy', use_semantic=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',
                    'truck', 'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation'
                ],
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
        ],
        classes=[
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/nuscenes/',
        occ_size=[200, 200, 16],
        pc_range=[-50, -50, -5.0, 50, 50, 3.0],
        use_semantic=True),
    test=dict(
        type='CustomNuScenesOccDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadOccupancy', use_semantic=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',
                    'truck', 'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation'
                ],
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
        ],
        classes=[
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        occ_size=[200, 200, 16],
        pc_range=[-50, -50, -5.0, 50, 50, 3.0],
        use_semantic=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='LoadOccupancy', use_semantic=True),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
                'manmade', 'vegetation'
            ],
            with_label=False),
        dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../work_dirs/test_overfit'
load_from = 'pretrained/r101_dcn_fcos3d_pretrain.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
occ_size = [200, 200, 16]
use_semantic = True
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [100, 50, 25]
volume_w_ = [100, 50, 25]
volume_z_ = [8, 4, 2]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]
model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=[100, 50, 25],
        volume_w=[100, 50, 25],
        volume_z=[8, 4, 2],
        num_query=900,
        num_classes=17,
        conv_input=[512, 256, 256, 128, 128, 64, 64],
        conv_output=[256, 256, 128, 128, 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1, 2, 1, 2, 1, 2, 1],
        embed_dims=[128, 256, 512],
        img_channels=[512, 512, 512],
        use_semantic=True,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=[128, 256, 512],
            encoder=dict(
                type='OccEncoder',
                num_layers=[1, 3, 6],
                pc_range=[-50, -50, -5.0, 50, 50, 3.0],
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-50, -50, -5.0, 50, 50, 3.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=[128, 256, 512],
                                num_points=[2, 4, 8],
                                num_levels=1),
                            embed_dims=[128, 256, 512])
                    ],
                    feedforward_channels=[256, 512, 1024],
                    ffn_dropout=0.1,
                    embed_dims=[128, 256, 512],
                    conv_num=2,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm',
                                     'conv'))))))
find_unused_parameters = True
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
gpu_ids = range(0, 1)
