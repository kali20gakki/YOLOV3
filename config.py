#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2020/03/10 17:41:09
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = '0.82'
import uuid
import numpy as np
import time
import six
import math
import random
import paddle
import paddle.fluid as fluid
import logging
import xml.etree.ElementTree
import codecs
import json
from time import strftime, localtime

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from PIL import Image, ImageEnhance, ImageDraw

logger = None
# train_parameters = {
#     "data_dir": "../data/data4379/pascalvoc",
#     "train_list": "train_list.txt",
#     "eval_list": "val_list.txt",
#     "class_dim": -1,
#     "label_dict": {}, 
#     "num_dict": {},
#     "image_count": -1,
#     "continue_train": True,     # 是否加载前一次的训练参数，接着训练
#     "pretrained": False,        # 是否使用预训练模型
#     "pretrained_model_dir": "./pretrained-model",
#     "save_model_dir": "./yolo-model",
#     "model_prefix": "yolo-v3",
#     "freeze_dir": "freeze_model",
#     "use_tiny": False,          # 是否使用 裁剪 tiny 模型
#     "yolo_type": 'ShuffleNetV2_YOLOv3', # YOLO backbone DarkNet53_YOLOv3 or ShuffleNetV2_YOLOv3
#     "max_box_num": 20,          # 一幅图上最多有多少个目标
#     "num_epochs": 80,           # 轮数
#     "train_batch_size": 32,     # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉；如果使用 tiny，可以适当大一些
#     "use_gpu": True,
#     "yolo_cfg": {
#         "input_size": [3, 384, 384],    # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
#         # "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], #416
#         "anchors": [9, 12, 15, 28, 30, 21,  28, 56, 57, 42, 54, 110, 107, 83, 144, 183, 344, 301],#384
#         # "anchors": [8, 10, 12, 23, 25, 18, 23, 47, 48, 35, 45, 92, 89, 69, 120, 152, 287, 251],#320
#         "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#     },
#     "yolo_tiny_cfg": {
#         "input_size": [3, 384, 384],
#         # "anchors": [6, 8, 13, 15, 22, 34, 48, 50, 81, 100, 205, 191], #256
#         "anchors": [9, 12, 20,  23,  33, 51, 72, 75, 122, 150, 308, 287],
#         "anchor_mask": [[3, 4, 5], [0, 1, 2]]
#     },
#     "ignore_thresh": 0.7,
#     "mean_rgb": [127.5, 127.5, 127.5],
#     "mode": "train",
#     "multi_data_reader_count": 4,
#     "apply_distort": True,
#     "nms_top_k": 400,
#     "nms_pos_k": 100,
#     "valid_thresh": 0.005,
#     "nms_thresh": 0.45,
#     "image_distort_strategy": {
#         "expand_prob": 0.5,
#         "expand_max_ratio": 4,
#         "hue_prob": 0.5,
#         "hue_delta": 18,
#         "contrast_prob": 0.5,
#         "contrast_delta": 0.5,
#         "saturation_prob": 0.5,
#         "saturation_delta": 0.5,
#         "brightness_prob": 0.5,
#         "brightness_delta": 0.125
#     },
#     "sgd_strategy": {
#         "learning_rate": 0.002,
#         "lr_epochs": [10, 45, 80, 110, 135, 160, 180],
#         "lr_decay": [1, 0.5, 0.25, 0.1, 0.025, 0.004, 0.001, 0.0005]
#     },
#     "early_stop": {
#         "sample_frequency": 50,
#         "successive_limit": 10,
#         "min_loss": 0.00000005,
#         "min_curr_map": 0.84
#     }
# }


train_parameters = {
    "data_dir": "data",
    "train_list": "train_list.txt",
    "eval_list": "val_list.txt",
    "use_filter": False,
    "class_dim": -1,
    "label_dict": {},
    "num_dict": {},
    "image_count": -1,
    "continue_train": True,     # 是否加载前一次的训练参数，接着训练
    "pretrained": False,
    "pretrained_model_dir": "model/pretrained_model",
    "save_model_dir": "model/model",
    "model_prefix": "yolo-v3",
    "freeze_dir": "model/freeze_model",
    "use_tiny": False,          # 是否使用 裁剪 tiny 模型
    "yolo_type": 'DarkNet53_YOLOv3', # YOLO backbone DarkNet53_YOLOv3 or ShuffleNetV2_YOLOv3
    "max_box_num": 50,          # 一幅图上最多有多少个目标
    "num_epochs": 110,
    "train_batch_size": 16,      # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉
    "use_gpu": False,
    "yolo_cfg": {
        "input_size": [3, 384, 384],    # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
        # "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], #416
        "anchors": [9, 12, 15, 28, 30, 21,  28, 56, 57, 42, 54, 110, 107, 83, 144, 183, 344, 301],#384
        # "anchors": [8, 10, 12, 23, 25, 18, 23, 47, 48, 35, 45, 92, 89, 69, 120, 152, 287, 251],#320
        "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    },
    "yolo_tiny_cfg": {
        "input_size": [3, 384, 384],
        "anchors": [9, 12, 20,  23,  33, 51, 72, 75, 122, 150, 308, 287],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,
    "nms_top_k": 400,
    "nms_pos_k": 100,
    "valid_thresh": 0.005,
    "nms_thresh": 0.45,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.5
    },
    "sgd_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [10, 45, 80, 110, 135, 160, 180],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.025, 0.004, 0.001, 0.0005]
    },
    "early_stop": {
        "sample_frequency": 50,
        "rise_limit": 10,
        "min_loss": 0.00000005,
        "min_curr_map": 0.84
    }
}

def init_train_parameters():
    """
        初始化训练参数，主要是初始化图片数量，类别数
    """
    file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    label_list = os.path.join(train_parameters['data_dir'], 'label_list')
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['num_dict'][index] = line.strip()
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)
    return train_parameters




def init_log_config():
    """
        初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    name = 'train'+strftime("%Y_%m_%d-%H_%M", localtime()) + '.log'
    log_name = os.path.join(log_path, name)
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
    