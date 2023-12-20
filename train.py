import torch, torchvision
import mmaction
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmengine.utils.dl_utils import collect_env
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
from operator import itemgetter
from mmengine.runner import set_random_seed

import os.path as osp
import mmengine
from mmengine.runner import Runner


# # Choose to use a config and initialize the recognizer
# config = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
# config = Config.fromfile(config)
# # Setup a checkpoint file to load
# checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
# # Initialize the recognizer
# model = init_recognizer(config, checkpoint, device='cuda:0')

# video = 'mmaction2/demo/demo.mp4'
# label = 'mmaction2/tools/data/kinetics/label_map_k400.txt'
# results = inference_recognizer(model, video)

# pred_scores = results.pred_score.tolist()
# score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
# score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
# top5_label = score_sorted[:5]

# labels = open(label).readlines()
# labels = [x.strip() for x in labels]
# results = [(labels[k[0]], k[1]) for k in top5_label]

# print('The top-5 labels with corresponding scores are:')
# for result in results:
#     print(f'{result[0]}: ', result[1])
    
cfg = Config.fromfile('mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')


# Modify dataset type and path
cfg.data_root = 'mmaction2/kinetics400_tiny/train/'
cfg.data_root_val = 'mmaction2/kinetics400_tiny/val/'
cfg.ann_file_train = 'mmaction2/kinetics400_tiny/kinetics_tiny_train_video.txt'
cfg.ann_file_val = 'mmaction2/kinetics400_tiny/kinetics_tiny_val_video.txt'


cfg.test_dataloader.dataset.ann_file = 'mmaction2/kinetics400_tiny/kinetics_tiny_val_video.txt'
cfg.test_dataloader.dataset.data_prefix.video = 'mmaction2/kinetics400_tiny/val/'

cfg.train_dataloader.dataset.ann_file = 'mmaction2/kinetics400_tiny/kinetics_tiny_train_video.txt'
cfg.train_dataloader.dataset.data_prefix.video = 'mmaction2/kinetics400_tiny/train/'

cfg.val_dataloader.dataset.ann_file = 'mmaction2/kinetics400_tiny/kinetics_tiny_val_video.txt'
cfg.val_dataloader.dataset.data_prefix.video  = 'mmaction2/kinetics400_tiny/val/'


# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 2
# We can use the pre-trained TSN model
cfg.load_from = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# Set up working dir to save files and logs.
cfg.work_dir = 'mmaction2/tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // 16
cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // 16
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
cfg.train_cfg.max_epochs = 10

cfg.train_dataloader.num_workers = 2
cfg.val_dataloader.num_workers = 2
cfg.test_dataloader.num_workers = 2

# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')



# Create work_dir
mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

# build the runner from config
runner = Runner.from_cfg(cfg)

# start training
runner.train()