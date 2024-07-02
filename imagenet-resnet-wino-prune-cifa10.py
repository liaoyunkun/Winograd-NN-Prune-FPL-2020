#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing
import glob
import pickle

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

from winograd_conv import WinogradConv
from winograd_imtrans import WinogradImTrans

TOTAL_BATCH_SIZE = 128
# Modified Here
INPUT_SHAPE = 32
DEPTH = None
test = False
mask_dict = None
use_mask = False


class Model(ModelDesc):
    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    def _get_inputs(self):
        # uint8 instead of float32 is used as input type to reduce copy overhead.
        # It might hurt the performance a liiiitle bit.
        # The pretrained models were trained with float32.
        return [InputDesc(tf.uint8, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        # Modified Here
        
        # nomalization the input image
        image = tf.cast(image,tf.float32)*(1.0/128)
        
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(layername, l, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):

            l = Conv2D('conv1', image, 64, 7, stride=2, nl=tf.identity)
            # l = BatchNorm('conv1_bn', l)
            l = MaxPooling('pool1', l, 3, stride=2, padding='SAME')

            l_bra = BatchNorm('res2a_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_2a_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W2a_2a', l_bra, 64, 64, mask=mask_dict['Winograd_W2a_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res2a_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_2a_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W2a_2b', l_bra, 64, 64, mask=mask_dict['Winograd_W2a_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res2a_bn2c', l_bra)

            # l = tf.nn.relu(l)
            l = BNReLU('res2a_1_relu', l)
            l = Conv2D('res2a_1', l, 64, 1, nl=tf.identity)
            l = BatchNorm('res2a_bn1', l)
            l = l + l_bra

            l_bra = BatchNorm('res2b_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_2b_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W2b_2a', l_bra, 64, 64, mask=mask_dict['Winograd_W2b_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res2b_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_2b_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W2b_2b', l_bra, 64, 64, mask=mask_dict['Winograd_W2b_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res2b_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = l + l_bra
            l = MaxPooling('pool2', l, 3, stride=2, padding='SAME')

            l_bra = BatchNorm('res3a_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_3a_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W3a_2a', l_bra, 64, 128, mask=mask_dict['Winograd_W3a_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res3a_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_3a_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W3a_2b', l_bra, 128, 128, mask=mask_dict['Winograd_W3a_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res3a_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = Conv2D('res3a_1', l, 128, 1, nl=tf.identity)
            l = BatchNorm('res3a_bn1', l)
            l = l + l_bra

            l_bra = BatchNorm('res3b_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_3b_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W3b_2a', l_bra, 128, 128, mask=mask_dict['Winograd_W3b_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res3b_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_3b_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W3b_2b', l_bra, 128, 128, mask=mask_dict['Winograd_W3b_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res3b_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = l + l_bra
            l = MaxPooling('pool3', l, 3, stride=2, padding='SAME')

            l_bra = BatchNorm('res4a_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_4a_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W4a_2a', l_bra, 128, 256, mask=mask_dict['Winograd_W4a_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res4a_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_4a_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W4a_2b', l_bra, 256, 256, mask=mask_dict['Winograd_W4a_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res4a_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = Conv2D('res4a_1', l, 256, 1, nl=tf.identity)
            l = BatchNorm('res4a_bn1', l)
            l = l + l_bra

            l_bra = BatchNorm('res4b_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_4b_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W4b_2a', l_bra, 256, 256, mask=mask_dict['Winograd_W4b_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res4b_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_4b_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W4b_2b', l_bra, 256, 256, mask=mask_dict['Winograd_W4b_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res4b_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = l + l_bra
            # l = MaxPooling('pool4', l, 3, stride=2, padding='SAME')

            l_bra = BatchNorm('res5a_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_5a_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W5a_2a', l_bra, 256, 512, mask=mask_dict['Winograd_W5a_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res5a_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_5a_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W5a_2b', l_bra, 512, 512, mask=mask_dict['Winograd_W5a_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res5a_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = Conv2D('res5a_1', l, 512, 1, nl=tf.identity)
            l = BatchNorm('res5a_bn1', l)
            l = l + l_bra

            l_bra = BatchNorm('res5b_bn2a', l)
            l_bra = WinogradImTrans('WinogradImTrans_5b_2a', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W5b_2a', l_bra, 512, 512, mask=mask_dict['Winograd_W5b_2a/W'] if use_mask else None)
            l_bra = BatchNorm('res5b_bn2b', l_bra)
            l_bra = WinogradImTrans('WinogradImTrans_5b_2b', l_bra, tf.nn.relu)
            l_bra = WinogradConv('Winograd_W5b_2b', l_bra, 512, 512, mask=mask_dict['Winograd_W5b_2b/W'] if use_mask else None)
            l_bra = BatchNorm('res5b_bn2c', l_bra)

            l = tf.nn.relu(l)
            l = l + l_bra
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            l = Dropout('drop_fc', l, 0.85)
            # l = Dropout('drop_fc', l, 0.7)
            # Modified Here
            logits = FullyConnected('linear', l, 10, nl=tf.identity)
            tf.nn.softmax(logits, name='output')
        # Modified Here
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
    # Modified
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
# Modified Here,get Cifar10 data
def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config(fake=False, data_format='NHWC'):
    if fake:
        dataset_train = dataset_val = FakeData(
            [[64, 32, 32, 3], [64]], 1000, random=False, dtype='uint8')
    else:
        dataset_train = get_data('train')
        dataset_val = get_data('test')

    eval_freq = 5

    return TrainConfig(
        model=Model(data_format=data_format),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val,
            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
            [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        max_epoch=400 if not test else 1,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--pruned_dir', help='the directory of pruned model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NHWC')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument('--test', help='test the model')
    args = parser.parse_args()

    if args.test:
        test = True

    if args.pruned_dir:
        mask_file_dir = glob.glob(os.path.join(args.pruned_dir, '*.pkl'))[0]
        use_mask = True

    DEPTH = args.depth
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    NR_GPU = get_nr_gpu()
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    logger.set_logger_dir(
        os.path.join('train_log', os.path.basename(__file__).split('.')[0]))
    logger.info("Running on {} GPUs. Batch size per GPU: {}".format(NR_GPU, BATCH_SIZE))
    config = get_config(fake=args.fake, data_format=args.data_format)
    if args.pruned_dir:
        model_file = glob.glob(os.path.join(args.pruned_dir, '*.data*'))[0]
        config.session_init = SaverRestore(model_file)
    if args.load:
        config.session_init = SaverRestore(args.load)
    if use_mask:
        mask_dict = pickle.load(open(mask_file_dir, 'rb'))
        print 'loading mask file: ', mask_file_dir
    #config.nr_tower = NR_GPU
    trainer = SyncMultiGPUTrainerParameterServer(NR_GPU, ps_device='gpu')
    launch_train_with_config(config, trainer)
