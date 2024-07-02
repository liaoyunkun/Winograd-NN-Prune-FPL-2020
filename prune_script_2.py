#!/usr/bin/python 
'''
set the prune threshold by total layer's statistic
layer_i's density = sum(layer_i)/sum(all layer)*density
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
    generate prune mask for a winograd layer in 
    a irregular-structural style
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
    for i in range(shape[0]):
        for j in range(in_channel):
            val = mat[i][j][:]
            # sort val
            sorted_index = np.argsort(val)
            for k in range(pruned_channel):
                pruned_index = sorted_index[k]
                mask[i][j][pruned_index] = 0
    return mask

def computeThreshold(weight_dict,mask_dict,total_prune_ratio):
    '''
    Sort all un-pruned weights and calculate the threshold T
    according to the total prune ratio
    @parameter
        weight_dict:{winograd layer name:winograd layer weight}
        mask_dict:{winograd layer name:prune mask}
            shape of winograd layer weight:(16,in_channel,out_channel)
        total_prune_ratio:the prune ratio for the whole model
    @return
        threshold value
    '''
    un_pruned = []
    for layerName,layerWeight in weight_dict.items():
        mask = mask_dict[layerName]
        layerWeight = weight_dict[layerName]
        shape0 = layerWeight.shape[0]
        in_channel = layerWeight.shape[1]
        out_Chanel = layerWeight.shape[2]
        for i in range(shape0):
            for j in range(in_channel):
                for k in range(out_Chanel):
                    if(mask[i][j][k] == 1):
                        # un-pruned weights
                        # need absolute weight to show importance!
                        un_pruned.append(abs(layerWeight[i][j][k]))
    un_pruned = np.array(un_pruned)
    sort_index = np.argsort(un_pruned)
    thres_index = sort_index[int(len(un_pruned)*total_prune_ratio)]
    thres_val = un_pruned[thres_index]
    return thres_val

def computePruneRatio(thres_val,layerWeight,mask):
    '''
    calculate layer pruning ratio according to thres_val
    @parameter
        thres_val:threshold value for pruning
        layerWeigh:(16,in_channel,out_channel) ndarray
        mask:needed?
    @return
        prune_ratio:prune ratio for current layer
    @questions
        1. calculate prune ratio for all weights by 
            comparing it to thres_val
        2. calculate prune ratios for all structural block
            defined by us,and select the minimum prune ratio
            as the final prune ratio?
        + implement method 1 first
    '''
    # syntax sugar :-)
    # 1: < thres_val,prunable weights
    # 0: unprunable weights
    mask_tmp = (np.absolute(layerWeight)<thres_val).astype(int)
    prune_ratio = np.sum(mask_tmp)/float(mask_tmp.size)
    return prune_ratio

def PruningLayer(layerWeight,prune_ratio_temp):
    '''
    Apply Structural Irregualr Pruning
    @return
        mask:1 for keep,0 for prune
    '''
    density = 1-prune_ratio_temp
    return gen_prune_mask(layerWeight,density)




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
    # computer prune ratio for each layer
        total_prune_ratio = 1-density
        max_prune_ratio_dict = {}
        pruneRatio_dict = {}
        with tf.Session() as sess_pruneRatio:
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
	    #chpt = tf.train.latest_checkpoint(original_dir)
            model_path = os.path.join(weight_dir,"model-1160000")
	    saver.restore(sess_pruneRatio, model_path)
	    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            weight_dict = {}
            for v in all_vars:
                v_ = sess_pruneRatio.run(v)
                s = v.name 
                s = s[:s.find(':0')]
                if 'Winograd_W' in s and '/W' in s: 
                    weight_dict[s] = v_
        mask_dict = {}
        all_weights_num = 0
        # initialize mask_dict,max_prune_ratio_dict
        for layerName,layerWeight in weight_dict.items():
            mask_dict[layerName] = np.ones(layerWeight.shape)
            max_prune_ratio_dict[layerName] = 0.9
            all_weights_num += layerWeight.size
        prune_ratio_real = 0
        L_prunable = weight_dict.keys()
        pruned_layer = []
        pruned_weights_num = 0
        # record total pruned weights for each layer
        pruned_weights_num_dict = {}
        # Iterative Pruning for the whole model
        while((prune_ratio_real < total_prune_ratio) and (len(L_prunable) > 0)):
            # Sort all un-pruned weights and calculate the threshold T
            # according to the total_prune_ratio
            thres_val = computeThreshold(weight_dict,mask_dict,total_prune_ratio)
            for layerName in L_prunable:
                # Calculated layer pruning ratio according to thres_val
                # 
                layerWeight = weight_dict[layerName]
                mask = mask_dict[layerName]
                prune_ratio_temp = computePruneRatio(thres_val,layerWeight,mask)
                if(prune_ratio_temp > max_prune_ratio_dict[layerName]):
                    # remove current layer from L_prunable
                    L_prunable.remove(layerName)
                    pruned_layer.append(layerName)
                    prune_ratio_temp = max_prune_ratio_dict[layerName]
                # update mask for current layer
                #mask_dict[layerName] = PruningLayer(layerWeight,prune_ratio_temp)
                mask_dict[layerName] = gen_prune_mask(layerWeight,1.0-prune_ratio_temp)
            # calculate the number of pruned weights
            pruned_weights_num = 0
            for l in mask_dict.keys():
                pruned_weights_num += (mask_dict[l].size-np.sum(mask_dict[l]))
            # update prune_ratio_real
            prune_ratio_real = pruned_weights_num/all_weights_num
            print('debug:prune_ratio_real{}'.format(prune_ratio_real))
        # calculate density for each layer
        density_dict = {}
        for layerName,layerMask in mask_dict.items():
            density_dict[layerName] = np.sum(layerMask)/layerMask.size   
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
		# chpt = tf.train.latest_checkpoint(original_dir)
                model_path = os.path.join(weight_dir,"model-1160000")
		saver.restore(sess, model_path)
		all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		prune_mask_dict = {}
		i = 0
		for v in all_vars:
			v_ = sess.run(v)
			s = v.name
			s = s[:s.find(':0')]
			if 'Winograd_W' in s and '/W' in s:  
				# if args.apply_mask is None:
				print 'inserting ', s, 'with density of', density_dict[s]
				if args.existing_mask:
					mask = gen_prune_mask(v_ * existing_masks[s], float(density_dict[s]))
				else:
					mask = gen_prune_mask(v_, float(density_dict[s]))
				prune_mask_dict[s] = mask
				# else:
				# 	mask = apply_masks[s]
				sess.run(v.assign(v_ * mask))
				i += 1
        # prune
		saver.save(sess, args.output_dir + '/pruned_' + args.density + '/pruned_' + args.density)
	with open(out_dir + '/pruned_' + args.density + '/prune_mask_' + args.density + '.pkl', 'wb') as handle:
		pickle.dump(prune_mask_dict, handle)
		# os.system('cp %s/checkpoint.bak %s/checkpoint'%(original_dir, original_dir))
		# os.system('echo \'model_checkpoint_path: \"model-37830\"\' > %s/prune_chpt'%(exp_dir))
	print 'finished'
