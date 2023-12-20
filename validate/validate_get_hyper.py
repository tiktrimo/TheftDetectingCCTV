# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import cv2
import numpy as np
from collections import deque
from mmaction.apis.inferencers import MMAction2Inferencer
import matplotlib.pyplot as plt
import time
from os import listdir, makedirs, path
from os.path import isfile, join



config_path = '/home/hsm/Python/MMACTION2/validate/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = '/home/hsm/Python/MMACTION2/validate/tsm_best_acc_epoch7.pth' # can be a local path
video_path = '/home/hsm/Python/MMACTION2/validate/C_3_12_2_BU_SMC_08-07_13-31-31_CA_RGB_DF2_M1.mp4'   # you can specify your own picture path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, default='', help='Input video device')
    parser.add_argument(
        '--vid-out-dir',
        type=str,
        default='',
        help='Output directory of videos.')
    parser.add_argument(
        '--rec',
        type=str,
        default=config_path,
        help='Pretrained action recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=checkpoint_path,
        help='Path to the custom checkpoint file of the selected recog model. '
        'If it is not specified and "rec" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--label-file', type=str, default=None, help='label file for dataset.')
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the video in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--pred-out-file',
        type=str,
        default='',
        help='File to save the inference results.')

    call_args = vars(parser.parse_args())

    init_kws = ['rec', 'rec_weights', 'device', 'label_file']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def get_ratio(model, call_args, source_path, SEQUENCE_LENGTH, suffix):
    videos= [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".mp4" in f]
    videos = videos[:5]
    
    steal_count = 0
    for progress, video in enumerate(videos):
        cap = cv2.VideoCapture(f"{source_path}{video}")
        frames = deque(maxlen=SEQUENCE_LENGTH)
        plot = None
        steal_acc = 0
        result_file = open(f"/home/hsm/Python/MMACTION2/validate/results/S{SEQUENCE_LENGTH}_{video[:-4]}_{suffix}.txt", "w")
        while True:
            isValidCapture, frame = cap.read()
            if not isValidCapture: break
            
            frames.append(frame)
            if len(frames) != SEQUENCE_LENGTH: continue

            call_args["inputs"] = np.array(frames)
            results = model(**call_args)
            
            preds_none, preds_steal = results["predictions"][0]["rec_scores"][0]   
            ############################# Logging ##############################
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[::1,::1]
            # if plot is None: plot = plt.imshow(image)
            # else : plot.set_data(image)
            # plt.pause(0.1)
            # plt.draw()
            # print(preds_none, preds_steal)
            result_file.write(f"{preds_none:.5f} {preds_steal:.5f}\n")
        result_file.close()
            

        print(f"P:{progress + 1}/{len(videos)} S:{steal_count}/{progress + 1} C:{source_path}")
            
    return
def main():
    init_args, call_args = parse_args()
    init_args["input_format"] = "array"
    model = MMAction2Inferencer(**init_args)
    
    for sequence_length in range(10, 60, 5):
        get_ratio(model, call_args, "/home/hsm/Python/MMACTION2/validate/VS_12STEAL/", sequence_length, "STEAL")
        get_ratio(model, call_args, "/home/hsm/Python/MMACTION2/validate/VS_7FALL/", sequence_length, "NONE")
        get_ratio(model, call_args, "/home/hsm/Python/MMACTION2/validate/VS_9FIRE/", sequence_length, "NONE")
        get_ratio(model, call_args, "/home/hsm/Python/MMACTION2/validate/VS_10SMOKE/", sequence_length, "NONE")
        get_ratio(model, call_args, "/home/hsm/Python/MMACTION2/validate/VS_13FIGHT/", sequence_length, "NONE")


if __name__ == '__main__':
    main()