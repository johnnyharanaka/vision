def write_faster(TYPE, FOLD, EPOCH):
    HOME = '/mmdetection'
    FASTER_RCNN_CONFIG_PATH = f'{HOME}/mmdetection/configs/faster_rcnn/fire_faster_rcnn.py'

    FASTER_RCNN_CONFIG = f"""
_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

'''
# setting visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
        init_kwargs={{'entity': "johnny_eidi-ufms-universidade-federal-de-mato-grosso-do-sul", # The entity used to log on Wandb
                        'project': "Faster_RCNN", # Project name in WandB
                        }})
] 
visualizer = dict(
    #type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
    type="Visualizer", vis_backends=[dict(type="WandbVisBackend")]
    )
'''
data_root = 'data/Fire/Fold_{FOLD}/'
metainfo = {{
    'classes': ('Corn', ),
    'palette': [
        (220, 20, 60),
    ]
}}

num_classes = 1

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1
    ))

train_dataloader = dict(
    #batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='Train_Images/BBoxJson/fold_{FOLD}_train_{TYPE}.json',
        data_prefix=dict(img='Train_Images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='Val_Images/BBoxJson/fold_{FOLD}_val_{TYPE}.json',
        data_prefix=dict(img='Val_Images/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='Test_Images/BBoxJson/fold_{FOLD}_test_{TYPE}.json',
        data_prefix=dict(img='Test_Images/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'Val_Images/BBoxJson/fold_{FOLD}_val_{TYPE}.json')
test_evaluator = dict(ann_file=data_root + 'Test_Images/BBoxJson/fold_{FOLD}_test_{TYPE}.json')
pretrained = 'https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_b_1k_224.pth'
model = dict(
    data_preprocessor=dict(batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='FlashInternImage',
        core_op='DCNv4',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.,
        drop_path_rate=0.3,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=0.5,
        post_norm=True,
        with_cp=True,
        dw_kernel_size=3,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    #backbone=dict(
    #    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
    #    stage_with_dcn=(False, True, True, True)),
    #backbone=dict(
    #    type='rcvit_t',
    #    style='pytorch',
    #    fork_feat=True,
    #    drop_path_rate=0.1,
    #    init_cfg=dict(
    #        type='Pretrained',
    #        checkpoint='./mmdet/casvit/cas-vit-t.pth',
    #    )
    #),
    neck=dict(
      type='PAFPN',
      in_channels=[112, 224, 448, 896],
      #in_channels=[256, 512, 1024, 2048],
      #in_channels=[96, 128, 256, 512],
      #in_channels=[64, 96, 192, 384],
      #in_channels=[48, 64, 128, 256],
      out_channels=256,
      num_outs=5),
    #roi_head=dict(
    #    bbox_head=dict(
    #        loss_clus=dict(loss_weight=0.1, type='ClusterLoss'),
    #        num_classes=num_classes)),
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            num_classes=num_classes,
            loss_clus=dict(loss_weight=0.1, type='ClusterLoss'))),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(add_gt_as_proposals=True))))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={EPOCH}, val_interval=1)
# optimizer
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
#optimizer_config = dict(grad_clip=None)
#optim_wrapper = dict(
#    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
#    type='OptimWrapper')
"""

    # Salvando o arquivo
    with open(FASTER_RCNN_CONFIG_PATH, 'w') as file:
        file.write(FASTER_RCNN_CONFIG)

# Definindo os valores das variáveis
fold = 3
sizetype = 20
epoch = 15
write_faster(sizetype, fold, epoch)
