from mmengine import Config


import os.path as osp
import mmengine
from mmengine.runner import Runner

    
cfg = Config.fromfile('configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')


# Modify dataset type and path
cfg.data_root = 'cvs/train/'
cfg.data_root_val = 'cvs/val/'
cfg.ann_file_train = 'cvs/action_small_train_video.txt'
cfg.ann_file_val = 'cvs/action_small_val_video.txt'


cfg.test_dataloader.dataset.ann_file = 'cvs/action_small_val_video.txt'
cfg.test_dataloader.dataset.data_prefix.video = 'cvs/val/'

cfg.train_dataloader.dataset.ann_file = 'cvs/action_small_train_video.txt'
cfg.train_dataloader.dataset.data_prefix.video = 'cvs/train/'

cfg.val_dataloader.dataset.ann_file = 'cvs/action_small_val_video.txt'
cfg.val_dataloader.dataset.data_prefix.video  = 'cvs/val/'


# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 2
# We can use the pre-trained TSN model
cfg.load_from = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# Set up working dir to save files and logs.
cfg.work_dir = 'tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // 16
cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // 16
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
cfg.train_cfg.max_epochs = 10

cfg.train_dataloader.num_workers = 2
cfg.val_dataloader.num_workers = 2
cfg.test_dataloader.num_workers = 2


# Create work_dir
mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

# build the runner from config
runner = Runner.from_cfg(cfg)

# start training
runner.train()