#!/usr/bin/python 

# -*- coding: UTF-8 -*-
import numpy as np 
import tensorflow as tf 
import argparse 
import os 

from tensorpack import * 
from tensorpack.tfutils.symbolic_functions import * 
from tensorpack.tfutils.summary import * 
from tensorflow.contrib.layers import variance_scaling_initializer 

# import winograd3x3  
import winograd2x2_conv.winograd2x2_conv
import winograd2x2_imTrans.winograd2x2_imTrans

import pickle
import glob
import sys

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--meta_dir', help='The directory of the original model meta data.')
        parser.add_argument('--weight_dir',help='The directory of the saved model weights')
	parser.add_argument('--output_dir', help='The directory of the output model.')
	parser.add_argument('--gpu', help='GPU that will be used.')
        parser.add_argument('--existing_mask',help='load existing mask.')
	args = parser.parse_args()

	meta_dir = args.meta_dir
        # find the meta file
	meta_file = glob.glob(meta_dir + '/graph-*')[0]
	meta_file_name = os.path.basename(meta_file)
        out_dir = args.output_dir
        # configure GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.existing_mask:
		existing_masks = pickle.load(open(args.existing_mask, 'rb'))
	# if args.apply_mask:
	# 	apply_masks = pickle.load(open(args.apply_mask, 'rb'))
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
		modelpath = os.path.join(args.weight_dir,'model-1160000')
                saver.restore(sess, modelpath)
		all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                wino_layer_dict = {}
		prune_mask_dict = {}
		for v in all_vars:
			v_ = sess.run(v)
			s = v.name
			s = s[:s.find(':0')]
			print s
			if 'Winograd_W' in s and '/W' in s:  
				if args.existing_mask:
                                    wino_layer_dict[s] = v_ * existing_mask[s]
				else:
                                    wino_layer_dict[s] = v_
	with open(out_dir + '/golen_model' + '.pkl', 'wb') as handle:
		pickle.dump(wino_layer_dict, handle)
		# os.system('cp %s/checkpoint.bak %s/checkpoint'%(original_dir, original_dir))
		# os.system('echo \'model_checkpoint_path: \"model-37830\"\' > %s/prune_chpt'%(exp_dir))
	print 'finished'
