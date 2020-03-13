#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   img_utils.py
@Time    :   2020/03/10 18:36:00
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib
import numpy as np
import config
import random
import os
from PIL import Image, ImageEnhance
import xml
import cv2

train_parameters = config.init_train_parameters()

def box_to_center_relative(box, img_height, img_width):
    """
    将COCO数据集的标注框格式[x1, y1, w, h]转换成中心坐标模式[center_x, center_y, w, h]
    将绝对坐标值除以图片的H,W 归一化
    """
    assert len(box) == 4, "box should be a len(4) list or tuple"
    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w - 1, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, img_height - 1)

    x = (x1 + x2) / 2 / img_width
    y = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return np.array([x, y, w, h])


def resize_img(img, input_size):
    """
        缩放图像size
    """
    target_size = input_size
    # Image.BILINEAR ： 双线性
    # Image.NEAREST ：低质量
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img



def distort_image(img):
    """
        图像处理，用于数据集增广
    """
    def random_brightness(img):
        """
            随机亮度调整
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['brightness_prob']:
            brightness_delta = train_parameters['image_distort_strategy']['brightness_delta']
            delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)
        return img

    def random_contrast(img):
        """
            随机对比度调整
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['contrast_prob']:
            contrast_delta = train_parameters['image_distort_strategy']['contrast_delta']
            delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)
        return img
    
    def random_saturation(img):
        """
            随机饱和度调整
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['saturation_prob']:
            saturation_delta = train_parameters['image_distort_strategy']['saturation_delta']
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)
        return img

    def random_hue(img):
        """
            随机色调整
        """
        prob = np.random.uniform(0, 1)
        if prob < train_parameters['image_distort_strategy']['hue_prob']:
            hue_delta = train_parameters['image_distort_strategy']['hue_delta']
            delta = np.random.uniform(-hue_delta, hue_delta)
            img_hsv = np.array(img.convert('HSV'))
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
            img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
        return img
        
    ops = [random_brightness, random_contrast, random_saturation, random_hue]
    # 打乱处理步骤 
    np.random.shuffle(ops) 
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = ops[3](img)
    return img



def box_iou_xywh(box1, box2):
    """
        计算iou
    """
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    """
        box crop
    """
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    """
        随机裁剪
    """
    if random.random() > 0.6:
        return img, boxes, labels
    if len(boxes) == 0:
        return img, boxes, labels

    if not constraints:
        constraints = [
                (0.1, 1.0),
                (0.3, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.9, 1.0),
                (0.0, 1.0)]

    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / float(w),
                crop_h /float(h)
                ]])

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img, crop_boxes, crop_labels
    return img, boxes, labels



def random_expand(img, gtboxes, keep_ratio=True):
    """
        随机扩张
    """
    if np.random.uniform(0, 1) < train_parameters['image_distort_strategy']['expand_prob']:
        return img, gtboxes

    max_ratio = train_parameters['image_distort_strategy']['expand_max_ratio']
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow -w)
    off_y = random.randint(0, oh -h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_parameters['mean_rgb'][i]

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return Image.fromarray(out_img), gtboxes


def random_flip(img, gtboxes, thresh=0.5):
    """
        随机翻转
    """
    if random.random() > thresh:
        # img = img[:, ::-1, :]
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes


def random_interp(img, size, interp=None):
    """
        随机差值
    """
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    img = np.asarray(img)
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size[2] / float(w)
    im_scale_y = size[1] / float(h)
    img = cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    img = Image.fromarray(img)
    return img


def shuffle_gtbox(gtbox, gtlabel):
    """
        shuffle ground turth box
    """
    gt = np.concatenate(
        [gtbox, gtlabel[:, np.newaxis]], axis=1)

    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]

    return gt[:, :4], gt[:, 4]