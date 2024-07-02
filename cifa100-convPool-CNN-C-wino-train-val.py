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
        # nomalization the input image
        image = tf.cast(image,tf.float32)*(1.0/128)
        
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])
        # TODO:convPOOL-CNN-C-Winograd-Relu Network
        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            l = WinogradImTrans('WinogradImTrans_0',image,tf.nn.relu)
            l = WinogradConv('WinogradConv_0',l,3,96,mask=mask_dict['WinogradConv_0/W'] if use_mask else None)
            l = WinogradImTrans('WinogradImTrans_1',l,tf.nn.relu)
            l = WinogradConv('WinogradConv_1',l,96,96,mask=mask_dict['WinogradConv_1/W'] if use_mask else None)
            l = WinogradImTrans('WinogradImTrans_2',l,tf.nn.relu)
            l = WinogradConv('WinogradConv_2',l,96,96,mask=mask_dict['WinogradConv_2/W'] if use_mask else None)
            l = MaxPooling('Pool0',l,3,stride=2,padding='SAME')
            l = WinogradImTrans('WinogradImTrans_3',l,tf.nn.relu)
            l = WinogradConv('WinogradConv_3',l,96,192,mask=mask_dict['WinogradConv_3/W'] if use_mask else None)
            l = WinogradImTrans('WinogradImTrans_4',l,tf.nn.relu)
            l = WinogradConv('WinogradConv_4',l,192,192,mask=mask_dict['WinogradConv_4/W'] if use_mask else None)
            l = WinogradImTrans('WinogradImTrans_5',l,tf.nn.relu)
            l = WinogradConv('WinogradConv_5',l,192,192,mask=mask_dict['WinogradConv_5/W'] if use_mask else None)
            l = MaxPooling('Pool1',l,3,stride=2,padding='SAME')
            l = WinogradImTrans('WinogradImTrans_6',l,tf.nn.relu)
            l = WinogradConv('WinogradConv_6',l,192,192,mask=mask_dict['WinogradConv_6/W'] if use_mask else None)
            l = Conv2D('Conv2D_0',l,192,1,nl=tf.identity)
            l = tf.nn.relu(l)
            l = Conv2D('Conv2D_1',l,100,1,nl=tf.identity)
            logits = GlobalAvgPooling('GAP',l)
            tf.nn.softmax(logits,name='output')
        # define cost function
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
    # use Cifar100 dataset interface
    ds = dataset.Cifar100(train_or_test)
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

def eval_on_CIFAR100(model_file,data_dir):
    ds = get_data('test')
    pred_config = PredictConfig(
        model = Model(),
        session_init = get_model_loader(model_file),
        input_names=['input','label'],
        output_names=['wrong_vector']
    )
    pred = SimpleDatasetPredictor(pred_config,ds)
    acc1 = RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(),batch_size)
    print("Top1 Error: {}".format(acc1.ratio))

def get_config(fake=False, data_format='NHWC'):
    if fake:
        dataset_train = dataset_val = FakeData(
            [[64, 32, 32, 3], [64]], 1000, random=False, dtype='uint8')
    else:
        dataset_train = get_data('train')
        dataset_val = get_data('test')

    eval_freq = 5

    # TODO:need adjustment
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