#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   DarkNet53_YOLOv3.py
@Time    :   2020/03/10 17:34:24
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib
"""
训练基于DarkNet53的YOLOv3网络，目标检测
"""
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


class YOLOv3(object):
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num

        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'DarkNet53_YOLOv3'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                use_cudnn=True):
        '''
            DarkNet53里基本卷积块：Conv + BN + LeakyReLU
        '''
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        out = fluid.layers.batch_norm( 
            input=conv, act=None, 
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
        # 在batch_norm中使用 leaky 的话，只能使用默认的 alpha=0.02；如果需要设值，必须提出去单独来
        out = fluid.layers.leaky_relu(out, 0.1)
        return out

    def downsample(self, input, num_filters, filter_size=3, stride=2, padding=1):
        '''
            利用卷积层下采样
            卷积核固定
            size：3*3
            stride：2
            padding: 1
        '''
        self.downsample_ratio *= 2
        return self.conv_bn(input, 
                num_filters=num_filters, 
                filter_size=filter_size, 
                stride=stride, 
                padding=padding)

    def basicblock(self, input, num_filters):
        '''
            基本残差块
            输入x经过两层卷积，然后接第二层卷积的输出和输入x相加
        '''
        conv1 = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        conv2 = self.conv_bn(conv1, num_filters * 2, filter_size=3, stride=1, padding=1)
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self, input, num_filters, count):
        '''
            一组残差模块，由若干个残差模块堆积而成。
            每组中第一个残差模块滑动窗口大小与其他可以不同，
            以用来减少特征图在垂直和水平方向的大小
        '''
        res_out = self.basicblock(input, num_filters)
        for j in range(1, count):
            res_out = self.basicblock(res_out, num_filters)
        return res_out

    def upsample(self, input, scale=2):
        '''
            上采样：对输入图片进行大小调整，
            在高度方向宽度方向进行最邻近插值
            这里固定高宽的乘数因子scale为2
        '''
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out
    
    def yolo_detection_block(self, input, num_filters):
        '''
            YOLO 检测层
        '''
        assert num_filters % 2 == 0, "num_filters {} cannot be divided by 2".format(num_filters)
        conv = input
        for j in range(2):
            conv = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
            conv = self.conv_bn(conv, num_filters * 2, filter_size=3, stride=1, padding=1)
        route = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)

        # route用于上采样后融合
        # tip 为yolo_detection_block 最后一次卷积输出，再经过一次[1*1 36]卷积即可得到Pred
        return route, tip

    def net(self, img): 
        '''
            构建YOLOV3 Model
        '''
        # DarkNet53的结构基本卷积块的次数
        stages = [1,2,8,8,4]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256

        # 第一层Conv  [416, 416, 3] -> [416, 416, 32]
        conv1 = self.conv_bn(img, num_filters=32, filter_size=3, stride=1, padding=1)

        # 第一次下采样  [416, 416, 32] -> [208, 208, 64]
        downsample_  = self.downsample(conv1, conv1.shape[1] * 2)

        # 根据stage 组装剩下的模块：卷积残差+下采样
        blocks = []
        for i, stage_count in enumerate(stages):
            block = self.layer_warp(downsample_, 32 *(2**i), stage_count)
            blocks.append(block)
            if i < len(stages) - 1:
                downsample_ = self.downsample(block, block.shape[1]*2)

        # 取倒数三层，并且逆序，后面跨层级联需要
        # C0, C1, C2
        blocks = blocks[-1:-4:-1]

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self.yolo_detection_block(block, num_filters=512 // (2**i))
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),      # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 256//(2**i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs




class YOLOv3Tiny(object):
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num

        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'YOLOv3-tiny'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                num_groups=1,
                use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            groups=num_groups,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        out = fluid.layers.batch_norm(
            input=conv, act='relu', 
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))

        return out

    def depthwise_conv_bn(self, input, filter_size=3, stride=1, padding=1):
        num_filters = input.shape[1]
        return self.conv_bn(input, 
                num_filters=num_filters, 
                filter_size=filter_size, 
                stride=stride, 
                padding=padding, 
                num_groups=num_filters)

    def downsample(self, input, pool_size=2, pool_stride=2):
        self.downsample_ratio *= 2
        return fluid.layers.pool2d(input=input, pool_type='max', pool_size=pool_size,
                                    pool_stride=pool_stride)

    def basicblock(self, input, num_filters):
        conv1 = self.conv_bn(input, num_filters, filter_size=3, stride=1, padding=1)
        out = self.downsample(conv1)
        return out


    def upsample(self, input, scale=2):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out
    
    def yolo_detection_block(self, input, num_filters):
        route = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img): 
        # darknet-tiny
        stages = [16, 32, 64, 128, 256, 512]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256
        tmp = img
        blocks = []
        for i, stage_count in enumerate(stages):
            if i == len(stages) - 1:
                block = self.conv_bn(tmp, stage_count, filter_size=3, stride=1, padding=1)
                blocks.append(block)
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.conv_bn(blocks[-1], stage_count * 2, filter_size=1, stride=1, padding=0)
                blocks.append(block)
            else:
                tmp = self.basicblock(tmp, stage_count)
                blocks.append(tmp)
        
        blocks = [blocks[-1], blocks[3]]

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            if i < 1:
                route, tip = self.yolo_detection_block(block, num_filters=256 // (2**i))
            else:
                tip = self.conv_bn(block, num_filters=256, filter_size=3, stride=1, padding=1)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),      # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2**i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs
        

def get_yolo(is_tiny, class_num, anchors, anchor_mask):
    if is_tiny:
        return YOLOv3Tiny(class_num, anchors, anchor_mask)
    else:
        return YOLOv3(class_num, anchors, anchor_mask)

