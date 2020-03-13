#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   reader.py
@Time    :   2020/03/10 18:52:47
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib
import numpy as np
import config
import paddle
from img_utils import *
train_parameters = config.init_train_parameters()

def preprocess(img, bbox_labels, input_size, mode):
    """
        图片预处理
    """
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)
    if mode == 'train':
        if train_parameters['apply_distort']:
            # 图片处理
            img = distort_image(img)
        img, gtboxes = random_expand(img, sample_labels[:, 1:5])
        img, gtboxes, gtlabels = random_crop(img, gtboxes, sample_labels[:, 0])
        img, gtboxes = random_flip(img, gtboxes, thresh=0.5)
        gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes
    # img = resize_img(img, sample_labels, input_size)
    img = random_interp(img, input_size)
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels



def custom_reader(file_list, data_dir,input_size, mode):
    """
        数据集读取生成器
    """
    def reader():
        """
            reader实现,根据具体数据集格式实现
        """
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                ######################  以下可能是需要自定义修改的部分   ############################
                image_path, label_path = line.split()
                image_path = os.path.join(data_dir, image_path)
                label_path = os.path.join(data_dir, label_path)
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # bbox 的列表，每一个元素为这样
                # layout: label | xmin | ymin | xmax | ymax | difficult
                bbox_labels = []
                root = xml.etree.ElementTree.parse(label_path).getroot()
                for object in root.findall('object'):
                    bbox_sample = []
                    # start from 1
                    bbox_sample.append(float(train_parameters['label_dict'][object.find('name').text]))
                    bbox = object.find('bndbox')
                    box = [float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(bbox.find('xmax').text) - float(bbox.find('xmin').text), float(bbox.find('ymax').text)-float(bbox.find('ymin').text)]
                    # print(box, img.size)
                    difficult = float(object.find('difficult').text)
                    bbox = box_to_center_relative(box, im_height, im_width)
                    # print(bbox)
                    bbox_sample.append(float(bbox[0]))
                    bbox_sample.append(float(bbox[1]))
                    bbox_sample.append(float(bbox[2]))
                    bbox_sample.append(float(bbox[3]))
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                ######################  可能需要自定义修改部分结束   ############################
                if len(bbox_labels) == 0:
                    continue
                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)
                # sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0: continue
                boxes = sample_labels[:, 1:5]
                lbls = sample_labels[:, 0].astype('int32')
                difficults = sample_labels[:, -1].astype('int32')
                max_box_num = train_parameters['max_box_num']
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_lbls = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_lbls[0: cope_size] = lbls[0: cope_size]
                ret_difficults[0: cope_size] = difficults[0: cope_size]
                
                yield img, ret_boxes, ret_lbls, ret_difficults
            elif mode == 'test':
                img_path = os.path.join(line)
                yield Image.open(img_path)

    return reader

def load_test_file(test_file_path):
    '''
        加载测试集
    '''
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
        for sample in range(len(lines)):
            image_path, label_path = lines[sample].split('\t')
            image_path = os.path.join(data_dir, image_path)
            label_path = os.path.join(data_dir, label_path)
            
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            im_width, im_height = img.size
            # layout: label | xmin | ymin | xmax | ymax | difficult
            # print(label_path[:-1])
            root = xml.etree.ElementTree.parse(label_path[:-1]).getroot()
            gt_label = []
            gt_boxes = []
            difficult = []
            gt_list = []
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                gt_label.append(float(train_parameters['label_dict'][object.find('name').text]))
                bbox = object.find('bndbox')
                gt_boxes.append([float(bbox.find('xmin').text)/im_width, float(bbox.find('ymin').text)/im_height,
                                float(bbox.find('xmax').text)/im_width, float(bbox.find('ymax').text)/im_height])
                difficult.append(float(object.find('difficult').text))
                gt_list.append([int(train_parameters['label_dict'][object.find('name').text]),
                                float(bbox.find('xmin').text), float(bbox.find('ymin').text),
                                float(bbox.find('xmax').text), float(bbox.find('ymax').text)])
            if len(gt_label) == 0:
                continue
            # print(np.array(gt_list).shape)
            # print(gt_list)
            img = cv2.imread(image_path)
            try:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                print(image_path)
                continue
            input_w, input_h = img.size[0], img.size[1]
            image_shape = np.array([input_h, input_w], dtype='int32')
            img = img.resize(yolo_config["input_size"][1:], Image.BILINEAR)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
            img -= 127.5
            img *= 0.007843
            img = img[np.newaxis, :]
            val_data.append([img, image_shape, gt_label, gt_boxes, difficult, image_path])
            val_data2.append([img, image_shape, gt_list])

def single_custom_reader(file_path, data_dir, input_size, mode):
    """
        单线程数据读取
    """
    file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    reader = custom_reader(images, data_dir, input_size, mode)
    reader = paddle.reader.shuffle(reader, train_parameters['train_batch_size'])
    reader = paddle.batch(reader, train_parameters['train_batch_size'])
    return reader


def multi_process_custom_reader(file_path, data_dir, num_workers, mode):
    """
        多线程数据读取
    """
    file_path = os.path.join(data_dir, file_path)
    readers = []
    images = [line.strip() for line in open(file_path)]
    n = int(math.ceil(len(images) // num_workers))
    image_lists = [images[i: i + n] for i in range(0, len(images), n)]
    for l in image_lists:
        readers.append(paddle.batch(custom_reader(l, data_dir, mode),
                                          batch_size=train_parameters['train_batch_size'],
                                          drop_last=True))
    return paddle.reader.multiprocess_reader(readers, False)