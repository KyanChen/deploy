log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='CS',
                 group='CS',
                 name='E20230313_0')
             )
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../pretrain/segformer_mit-b1_512x512_160k_ade20k_20220620_112037-c3f39e00.pth'

resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
num_classes = 3
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2],
        num_stages=4,
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/expand_data/datasets'
img_size = (512, 512)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

CLASSES = ('bg', 'cloud', 'snow')
PALETTE = [[0, 0, 0], [0, 0, 255], [0, 255, 0]]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_size, ratio_range=(0.7, 1.3)),
    dict(type='RandomCrop', crop_size=img_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ChannelShuffle', prob=0.3),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_GF_train1 = dict(
    type=dataset_type,
    # CLASSES=CLASSES,
    # PALETTE=PALETTE,
    img_dir=data_root+'/Levir_CS/train/',
    ann_dir=data_root+'/Levir_CS/train/',
    reduce_zero_label=False,
    img_suffix='.tif',
    seg_map_suffix='.png',
    pipeline=train_pipeline
)
dataset_GF_train2 = dict(
    type=dataset_type,
    # CLASSES=CLASSES,
    # PALETTE=PALETTE,
    img_dir=data_root+'/Levir_CS/valid/',
    ann_dir=data_root + '/Levir_CS/valid/',
    reduce_zero_label=False,
    img_suffix='.tif',
    seg_map_suffix='.png',
    pipeline=train_pipeline
)
dataset_GF_train3 = dict(
    type=dataset_type,
    # CLASSES=CLASSES,
    # PALETTE=PALETTE,
    img_dir=data_root+'/Levir_CS/test/',
    ann_dir=data_root + '/Levir_CS/test/',
    reduce_zero_label=False,
    img_suffix='.tif',
    seg_map_suffix='.png',
    pipeline=train_pipeline
)

dataset_SY_train = dict(
    type=dataset_type,
    # CLASSES=CLASSES,
    # PALETTE=PALETTE,
    img_dir=data_root+'/ZY_CS_train',
    ann_dir=data_root + '/ZY_CS_train',
    reduce_zero_label=False,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    pipeline=train_pipeline
)


train_dataset = [dataset_GF_train1, dataset_GF_train2, dataset_GF_train3, dataset_SY_train, dataset_SY_train]

val_dataset = dict(
    type=dataset_type,
    # CLASSES=CLASSES,
    # PALETTE=PALETTE,
    img_dir=data_root+'/ZY_CS_train',
    ann_dir=data_root + '/ZY_CS_train',
    reduce_zero_label=False,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    test_mode=True,
    pipeline=test_pipeline
)


batch_size = 48
num_works = 8
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_works,
    persistent_workers=True,
    shuffle=True,
    train=train_dataset,
    val=val_dataset,
    test=val_dataset,
    val_dataloader=dict(samples_per_gpu=batch_size, workers_per_gpu=8, shuffle=False),
    test_dataloader=dict(samples_per_gpu=batch_size, workers_per_gpu=8, shuffle=False)
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
