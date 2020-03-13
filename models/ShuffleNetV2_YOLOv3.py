#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   ShuffleNetV2_YOLOv3.py
@Time    :   2020/03/10 17:36:41
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib

"""
主干网络基于shufflenet,检测分支改造后的yolo3
"""
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


class YOLOv3(object):
    """
    定义模型
    """
    def __init__(self, class_num, anchors, anchor_mask, scale=1.5):
        """
            模型参数初始化
        """
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num
        self.scale = scale
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
        return 'ShuffleNetV2_YOLOv3'

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

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      use_cudnn=True,
                      if_act=True,
                      name=None):
        """
        主干网络的conv+bn
        """
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            # param_attr=ParamAttr(
            #     initializer=MSRA(), name=name + '_weights'),
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)
        out = int((input.shape[2] - 1) / float(stride) + 1)
        bn_name = name + '_bn'
        if if_act:
            return fluid.layers.batch_norm(
                input=conv,
                act='relu',
                param_attr=ParamAttr(name=bn_name + "_scale"),
                bias_attr=ParamAttr(name=bn_name + "_offset"),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
        else:
            return fluid.layers.batch_norm(
                input=conv,
                param_attr=ParamAttr(name=bn_name + "_scale"),
                bias_attr=ParamAttr(name=bn_name + "_offset"),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')

    def channel_shuffle(self, x, groups):
        """
            channel_shuffle
        """
        batchsize, num_channels, height, width = x.shape[0], x.shape[
            1], x.shape[2], x.shape[3]
        channels_per_group = num_channels // groups

        # reshape
        x = fluid.layers.reshape(
            x=x, shape=[batchsize, groups, channels_per_group, height, width])

        x = fluid.layers.transpose(x=x, perm=[0, 2, 1, 3, 4])

        # flatten
        x = fluid.layers.reshape(
            x=x, shape=[batchsize, num_channels, height, width])

        return x

    def inverted_residual_unit(self,
                               input,
                               num_filters,
                               stride,
                               benchmodel,
                               name=None):
        """
            inverted_residual_unit
        """
        assert stride in [1, 2], \
            "supported stride are {} but your stride is {}".format([1, 2], stride)

        oup_inc = num_filters // 2
        inp = input.shape[1]

        if benchmodel == 1:
            x1, x2 = fluid.layers.split(
                input,
                num_or_sections=[input.shape[1] // 2, input.shape[1] // 2],
                dim=1)

            conv_pw = self.conv_bn_layer(
                input=x2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv1')

            conv_dw = self.conv_bn_layer(
                input=conv_pw,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                use_cudnn=True,
                name='stage_' + name + '_conv2')

            conv_linear = self.conv_bn_layer(
                input=conv_dw,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv3')

            out = fluid.layers.concat([x1, conv_linear], axis=1)

        else:
            # branch1
            conv_dw_1 = self.conv_bn_layer(
                input=input,
                num_filters=inp,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=inp,
                if_act=False,
                use_cudnn=True,
                name='stage_' + name + '_conv4')

            conv_linear_1 = self.conv_bn_layer(
                input=conv_dw_1,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv5')

            # branch2
            conv_pw_2 = self.conv_bn_layer(
                input=input,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv1')

            conv_dw_2 = self.conv_bn_layer(
                input=conv_pw_2,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                use_cudnn=True,
                name='stage_' + name + '_conv2')

            conv_linear_2 = self.conv_bn_layer(
                input=conv_dw_2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                name='stage_' + name + '_conv3')
            out = fluid.layers.concat([conv_linear_1, conv_linear_2], axis=1)

        return self.channel_shuffle(out, 2), out

    def upsample(self, input, scale=2):
        """
        upsample
        """
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

    def yolo_detection_block(self, input, num_filters, k=2):
        """
        yolo_detection_block
        """
        assert num_filters % 2 == 0, "num_filters {} cannot be divided by 2".format(num_filters)
        conv = input
        for j in range(k):
            conv = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
            conv = self.conv_bn(conv, num_filters * 2, filter_size=3, stride=1, padding=1)
        route = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                use_cudnn=True):
        """
        检测分支的conv_bn
        """
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            use_cudnn=use_cudnn,
            # param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        # 在batch_norm中使用 leaky 的话，只能使用默认的 alpha=0.02；如果需要设值，必须提出去单独来
        out = fluid.layers.batch_norm(
            input=conv, act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
        out = fluid.layers.leaky_relu(out, 0.1)
        return out

    def net(self, input):
        """
        构造模型
        """
        scale = self.scale
        stage_repeats = [4, 8, 4]

        if scale == 0.25:
            stage_out_channels = [-1, 24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [-1, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("""{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))
        # conv1
        blocks = []
        input_channel = stage_out_channels[1]
        conv1 = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=input_channel,
            padding=1,
            stride=2,
            name='stage1_conv')

        self.downsample_ratio *= 2

        pool1 = fluid.layers.pool2d(
            input=conv1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        conv = pool1
        self.downsample_ratio *= 2

        # bottleneck sequences
        for idxstage in range(len(stage_repeats)):
            numrepeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    conv, out = self.inverted_residual_unit(
                        input=conv,
                        num_filters=output_channel,
                        stride=2,
                        benchmodel=2,
                        name=str(idxstage + 2) + '_' + str(i + 1))
                    self.downsample_ratio *= 2
                else:
                    conv, out = self.inverted_residual_unit(
                        input=conv,
                        num_filters=output_channel,
                        stride=1,
                        benchmodel=1,
                        name=str(idxstage + 2) + '_' + str(i + 1))
            blocks.append(conv)

        blocks = [blocks[-1], blocks[-2], blocks[-3]]

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
                # print(block.shape)
            
            route, tip = self.yolo_detection_block(block, num_filters=256 // (2 ** i), k=2)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),  # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs


class YOLOv3Tiny(object):
    """
    yolo_tiny
    """
    def __init__(self, class_num, anchors, anchor_mask):
        """
        初始化模型的超参数
        """
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
        """
        卷积 + bn
        :return:
        """
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
        """
        深度可分离卷积 + bn
        :return:
        """
        num_filters = input.shape[1]
        return self.conv_bn(input,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            num_groups=num_filters)

    def downsample(self, input, pool_size=2, pool_stride=2):
        """
        通过池化进行下采样
        :return:
        """
        self.downsample_ratio *= 2
        return fluid.layers.pool2d(input=input, pool_type='max', pool_size=pool_size,
                                   pool_stride=pool_stride)

    def basicblock(self, input, num_filters):
        """
        基础的卷积块
        :return:
        """
        conv1 = self.conv_bn(input, num_filters, filter_size=3, stride=1, padding=1)
        out = self.downsample(conv1)
        return out

    def upsample(self, input, scale=2):
        """
        上采样
        :return:
        """
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
        """
        yolo检测模块
        :return:
        """
        route = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img):
        """
        整体网络结构构建
        :return:
        """
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
                route, tip = self.yolo_detection_block(block, num_filters=256 // (2 ** i))
            else:
                tip = self.conv_bn(block, num_filters=256, filter_size=3, stride=1, padding=1)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),  # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs


def get_yolo(is_tiny, class_num, anchors, anchor_mask):
    """
    根据is_tiny来构建yolo网络或yolo_tiny
    :return:
    """
    if is_tiny:
        return YOLOv3Tiny(class_num, anchors, anchor_mask)
    else:
        return YOLOv3(class_num, anchors, anchor_mask)
