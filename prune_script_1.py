#!/usr/bin/python
'''
set the prune threshold by each layer's statistic
''' 

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

def gen_prune_mask(mat, density):
    '''
    @parameter
        mat: numpy ndarray (16,in_Channel,out_Chanel)
        density: float,prune ratio     
    @return 
        mask: shape shape as mat,0 indicate pruned weight
    '''
    shape = mat.shape
    in_channel = shape[1]
    out_channel = shape[2]
    mask = np.ones(shape)
    pruned_channel = int(out_channel*(1-density))
    # need absolute weight
    mat = np.absolute(mat)
    for i in range(shape[0]):
        for j in range(in_channel):
            val = mat[i][j][:]
            # sort val
            sorted_index = np.argsort(val)
            for k in range(pruned_channel):
                pruned_index = sorted_index[k]
                mask[i][j][pruned_index] = 0
    return mask

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--original_dir', help='The directory of the original model.')
	parser.add_argument('--output_dir', help='The directory of the output model.')
        parser.add_argument('--weight_dir',help='The directory of the saved model')
	parser.add_argument('--gpu', help='GPU that will be used.')
	parser.add_argument('--density', help='The density of each layer.')
	parser.add_argument('--existing_mask', help='The existing mask file.')
	args = parser.parse_args()

	original_dir = args.original_dir
        weight_dir = args.weight_dir
	meta_file = glob.glob(original_dir + '/graph-*')[0]
	meta_file_name = os.path.basename(meta_file)
	out_dir = args.output_dir
	# density = args.density.split(',')
	density = float(args.density)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# import pdb; pdb.set_trace()
	os.system('mkdir %s/pruned_%s' % (out_dir, args.density))
	os.system('cp %s/%s %s/pruned_%s/%s' % (original_dir, meta_file_name, out_dir, args.density, meta_file_name))
	if args.existing_mask:
		existing_masks = pickle.load(open(args.existing_mask, 'rb'))
	# if args.apply_mask:
	# 	apply_masks = pickle.load(open(args.apply_mask, 'rb'))
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
		#chpt = tf.train.latest_checkpoint(original_dir)
                weight_path = os.path.join(weight_dir,"model-1160000")
		saver.restore(sess, weight_path)
		all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		prune_mask_dict = {}
		i = 0
		for v in all_vars:
			v_ = sess.run(v)
			s = v.name
			s = s[:s.find(':0')]
			print s
			if 'Winograd_W' in s and '/W' in s:  
				# if args.apply_mask is None:
				print 'inserting ', s, 'with density of', density
				if args.existing_mask:
					mask = gen_prune_mask(v_ * existing_masks[s], float(density))
				else:
					mask = gen_prune_mask(v_, float(density))
				prune_mask_dict[s] = mask
				# else:
				# 	mask = apply_masks[s]
				sess.run(v.assign(v_ * mask))
				i += 1
		saver.save(sess, args.output_dir + '/pruned_' + args.density + '/pruned_' + args.density)
	with open(out_dir + '/pruned_' + args.density + '/prune_mask_' + args.density + '.pkl', 'wb') as handle:
		pickle.dump(prune_mask_dict, handle)
		# os.system('cp %s/checkpoint.bak %s/checkpoint'%(original_dir, original_dir))
		# os.system('echo \'model_checkpoint_path: \"model-37830\"\' > %s/prune_chpt'%(exp_dir))
	print 'finished'
