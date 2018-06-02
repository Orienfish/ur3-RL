####################################################################################
# This file is for testing the dqn learning model in the real environment
# Modified by xfyu on May 24
####################################################################################
# -*- coding: utf-8 -*-
# !/usr/bin/python
from __future__ import print_function

import tensorflow as tf
import cv2
import os
import sys
import random
import numpy as np
import realenv_test as env
import matplotlib.pyplot as plt
# import collect_code.pycontrol as ur

###################################################################################
# Important global parameters
###################################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
# specify the version of test model
VERSION = "testgrp5_nodrop_m"
TRAIN_DIR = "train_" + VERSION
TRAIN_DIR = os.path.join(PATH, TRAIN_DIR)
# the following files are all in training directories
READ_NETWORK_DIR = "saved_networks_part_" + VERSION
READ_NETWORK_DIR = os.path.join(TRAIN_DIR, READ_NETWORK_DIR)
# test result folder
TEST_DIR = "real_test_" + VERSION
TEST_DIR = os.path.join(PATH, TEST_DIR)
if not os.path.isdir(TEST_DIR):
	os.makedirs(TEST_DIR)
# used in pre-process the picture
RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128
# normalize the action
ACTION_NORM = 0.3*8.78
ANGLE_NORM = 100

# parameters used in testing
ACTIONS = 5 # number of valid actions
PAST_FRAME = 3
TEST_ROUND = 10

###################################################################################
# Functions
###################################################################################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def space_tiling(x): # expand from [None, 64] to [None, 4, 4, 64]
    x = tf.expand_dims(tf.expand_dims(x, 1), 1)
    return tf.tile(x, [1, 4, 4, 1])

'''
createNetwork - set the structure of CNN
'''
# network weights
W_conv1 = weight_variable([8, 8, PAST_FRAME, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([6, 6, 32, 64])
b_conv2 = bias_variable([64])

W_conv3 = weight_variable([4, 4, 128, 64])
b_conv3 = bias_variable([64])

W_conv4 = weight_variable([3, 3, 64, 64])
b_conv4 = bias_variable([64])

W_fc1 = weight_variable([256, 256])
b_fc1 = bias_variable([256])

W_fc2 = weight_variable([256, 256])
b_fc2 = bias_variable([256])

W_fc3 = weight_variable([256, ACTIONS])
b_fc3 = bias_variable([ACTIONS])

W_fc_info = weight_variable([PAST_FRAME*2, 64])
b_fc_info = bias_variable([64])

# input layer
# one state to train each time
s = tf.placeholder(dtype=tf.float32, name='s', shape=(None, RESIZE_WIDTH, RESIZE_HEIGHT, PAST_FRAME))
past_info = tf.placeholder(dtype=tf.float32, name='past_info', shape=(None, PAST_FRAME*2))
training = tf.placeholder_with_default(False, name='training', shape=())

# hidden layers
h_conv1 = conv2d(s, W_conv1, 4) + b_conv1
h_bn1 = tf.layers.batch_normalization(h_conv1, axis=-1, training=training, momentum=0.9)
h_relu1 = tf.nn.relu(h_bn1)
h_pool1 = max_pool_2x2(h_relu1) # [None, 16, 16, 32]

h_conv2 = conv2d(h_pool1, W_conv2, 2) + b_conv2
h_bn2 = tf.layers.batch_normalization(h_conv2, axis=-1, training=training, momentum=0.9)
h_relu2 = tf.nn.relu(h_bn2)
h_pool2 = max_pool_2x2(h_relu2) # [None, 4, 4, 64]

h_fc_info = tf.matmul(past_info, W_fc_info) + b_fc_info
h_bn_info = tf.layers.batch_normalization(h_fc_info, axis=-1, training=training, momentum=0.9)
h_relu_info = tf.nn.relu(h_bn_info) # [None, 64]

info_add = space_tiling(h_relu_info) # [None, 4, 4, 64]
layer3_input = tf.concat([h_pool2, info_add], 3) # [None, 4, 4, 128]
h_conv3 = conv2d(layer3_input, W_conv3, 1) + b_conv3
h_bn3 = tf.layers.batch_normalization(h_conv3, axis=-1, training=training, momentum=0.9)
h_relu3 = tf.nn.relu(h_bn3) # [None, 4, 4, 64]
# h_pool3 = max_pool_2x2(h_relu3) # [None, 2, 2, 64]

h_conv4 = conv2d(h_relu3, W_conv4, 1) + b_conv4
h_bn4 = tf.layers.batch_normalization(h_conv4, axis=-1, training=training, momentum=0.9)
h_relu4 = tf.nn.relu(h_bn4) # [None, 4, 4, 64]
h_pool4 = max_pool_2x2(h_relu4) # [None, 2, 2, 64]

h_pool4_flat = tf.reshape(h_pool4, [-1, 256]) # [None, 256]

h_fc1 = tf.matmul(h_pool4_flat, W_fc1) + b_fc1
h_bn_fc1 = tf.layers.batch_normalization(h_fc1, axis=-1, training=training, momentum=0.9)
h_relu_fc1 = tf.nn.relu(h_bn_fc1) # [None, 256]
    
h_fc2 = tf.matmul(h_relu_fc1, W_fc2) + b_fc2
h_bn_fc2 = tf.layers.batch_normalization(h_fc2, axis=-1, training=training, momentum=0.9)
h_relu_fc2 = tf.nn.relu(h_bn_fc2) # [None, 256]
# readout layer
readout = tf.matmul(h_relu_fc2, W_fc3) + b_fc3 # [None, 5]

'''
Neural Network Definitions --- not necessary in test
'''
'''
# define the cost function
a = tf.placeholder(dtype=tf.float32, name='a', shape=(None, ACTIONS))
y = tf.placeholder(dtype=tf.float32, name='y', shape=(None))
accuracy = tf.placeholder(dtype=tf.float32, name='accuracy', shape=())
# define cost
with tf.name_scope('cost'):
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    tf.summary.scalar('cost', cost)
with tf.name_scope('accuracy'):
    tf.summary.scalar('accuracy', accuracy)
# define training step
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(cost)
'''

'''
testNetwork - test the training performance, calculate the success rate

Input: s, action,readout
Return: success rate
'''
def testNetwork():
    # init the real test environment
    test_env = env.FocusEnv()
    action_space = test_env.actions
    endAngle = []
    '''
    Start tensorflow
    '''

    # saving and loading networks
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # load in half-trained networks
        checkpoint = tf.train.get_checkpoint_state(READ_NETWORK_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print("Could not find old network weights")
    	
	success_cnt = 0.0
	total_steps = 0.0
        # start test
	for test in range(TEST_ROUND):
	    init_angle, init_img_path = test_env.reset()
	                
	    # generate the first state, a_past is 0
	    img_t = cv2.imread(init_img_path)
	    img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
	    s_t = np.stack((img_t, img_t, img_t) , axis=2)
            action_t = np.stack((0.0, 0.0, 0.0), axis=0)
            angle_t = np.stack((init_angle/ANGLE_NORM, init_angle/ANGLE_NORM, init_angle/ANGLE_NORM), axis=0)
	    past_info_t = np.append(action_t, angle_t, axis=0)
	    step = 1
	    # start 1 episode
	    while True:
	        # run the network forwardly
	        readout_t = readout.eval(feed_dict={
			s : [s_t], 
			past_info : [past_info_t],
			training : False})[0]
        	print(past_info_t)
		print(readout_t)
		# determine the next action
		action_index = np.argmax(readout_t)
		a_input = action_space[action_index]
		# run the selected action and observe next state and reward
		angle_new, img_path_t1, terminal, success = test_env.step(a_input)

		if terminal:
			# save_last_pic(test, test_env.cur_state, test_env.dic[test_env.cur_state])
		        success_cnt += int(success) # only represents the rate of active terminate
		        total_steps += step
            	        endAngle.append(angle_new)
                        print("end at", angle_new)
			save_terminal_pic(test, angle_new, img_path_t1)
		        break
		            
		img_t1 = cv2.imread(img_path_t1)
		img_t1 = cv2.cvtColor(cv2.resize(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
		img_t1 = np.reshape(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT, 1)) # reshape, ready for insert
		angle_new = np.reshape(angle_new/ANGLE_NORM, (1,))
		action_new = np.reshape(a_input/ACTION_NORM, (1,))
		s_t1 = np.append(img_t1, s_t[:, :, :PAST_FRAME-1], axis=2)
		angle_t1 = np.append(angle_new, angle_t[:PAST_FRAME-1], axis=0)
		action_t1 = np.append(action_new, action_t[:PAST_FRAME-1], axis=0)
	        past_info_t1 = np.append(action_t1, angle_t1, axis=0)
		# print test info
		print("TEST EPISODE", test, "/ TIMESTEP", step, \
			"/ CURRENT ANGLE", test_env.cur_state, "/ ACTION", a_input)

		# update
		s_t = s_t1
		action_t = action_t1
		angle_t = angle_t1
		past_info_t = np.append(action_t, angle_t, axis=0)
		step += 1

    success_rate = success_cnt/TEST_ROUND
    step_cost = total_steps/TEST_ROUND
    print("success_rate:", success_rate, "step per episode:", step_cost)
    
    record_and_plot(endAngle)
    return success_rate

def TENG(img):
        guassianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        guassianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        return np.mean(guassianX * guassianX + 
                          guassianY * guassianY)

'''
record_and_plot
'''
def record_and_plot(endAngle):
    angle_path = os.path.join(TEST_DIR, "endAngle.txt")
    endFocus = []
    with open(angle_path, "w") as f:
        for i in range(TEST_ROUND):
            pic_name = str(i) + '.jpg'
            pic_path = os.path.join(TEST_DIR, pic_name)
            img = cv2.imread(pic_path)
            img = cv2.cvtColor(cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY) 
            endFocus.append(TENG(img))
            f.write(str(endAngle[i]) + ' ' + str(endFocus[i]) + '\n')
    
    plt.figure()
    plt.plot(endAngle, endFocus, 'bx')
    plt.savefig(os.path.join(TEST_DIR, "result"), dpi=1200)
    plt.show()

'''
save_terminal_pic - save the final picture and use the episode num and
		    the final angle to name it
'''
def save_terminal_pic(epi_num, angle_new, img_path_t1):
    img = cv2.imread(img_path_t1)
    # to avoid '.' appears in file name
    new_pic_name = str(epi_num) + '.jpg'
    new_pic_path = os.path.join(TEST_DIR, new_pic_name)
    print(new_pic_path)
    cv2.imwrite(new_pic_path, img)

if __name__ == '__main__':
	testNetwork()
