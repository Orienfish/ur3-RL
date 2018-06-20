####################################################################################
# This file is for testing the dqn learning model in the virtual environment
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
# import pycontrol as ur
import trainenv_f_action_v4 as env
import matplotlib.pyplot as plt

###################################################################################
# Important global parameters
###################################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
# tf.app.flags defined input parameters
# Necessary: VERSION, ENV_PATH.
# Annotate the parameters in training
tf.app.flags.DEFINE_string('TEST_PATH', '/home/robot/RL/data/new_grp2','test image path')
tf.app.flags.DEFINE_string('VERSION', 'virf_grp2_changepoint10', 'version of this training')
# tf.app.flags.DEFINE_string('BASED_VERSION', '', 'version of the based model')
tf.app.flags.DEFINE_string('ENV_PATH', 'trainenv_virf_v5', 'path of environment class file')
# tf.app.flags.DEFINE_integer('NUM_TRAINING_STEPS', 50000, 'number of time steps in one training')
# tf.app.flags.DEFINE_integer('OBSERVE', 1000, 'number of time steps to observe before training')
# tf.app.flags.DEFINE_integer('EXPLORE', 30000, 'number of time steps to explore after observation')
# tf.app.flags.DEFINE_integer('REPLAY_MEMORY', 500, 'number of previous transitions to remember')
# tf.app.flags.DEFINE_float('LEARNING_RATE', 0.001, 'learning rate for optimizer')
tf.app.flags.DEFINE_integer('TEST_ROUND', 1000, 'how many episodes in the test')
# tf.app.flags.DEFINE_float('GAMMA', 0.99, 'decay rate of past observations')
# tf.app.flags.DEFINE_integer('BATCH', 32, 'size of minibatch')
# tf.app.flags.DEFINE_float('FINAL_EPSILON', 0.001, 'final value of epsilon')
# tf.app.flags.DEFINE_float('INITIAL_EPSILON', 0.01, 'starting value of epsilon')
# tf.app.flags.DEFINE_integer('COST_RECORD_STEP', 100, 'cost recording step')
# tf.app.flags.DEFINE_integer('NETWORK_RECORD_STEP', 1000, 'network recording step')
# tf.app.flags.DEFINE_integer('REWARD_RECORD_STEP', 100, 'reward recording step')
# tf.app.flags.DEFINE_integer('STEP_RECORD_STEP', 100, 'step recording step')
# tf.app.flags.DEFINE_integer('SUCCESS_RATE_TEST_STEP', 1000, 'testing accuracy step')
tf.app.flags.DEFINE_float('PER_GPU_USAGE', 0.333, 'how much space taken per gpu')
tf.app.flags.DEFINE_integer('MAX_STEPS', 10, 'max steps defined in env')
tf.app.flags.DEFINE_float('MIN_ANGLE', 30.0, 'min angle defined in env')
tf.app.flags.DEFINE_float('MAX_ANGLE', 69.0, 'max angle defined in env')
FLAGS = tf.app.flags.FLAGS

# define global variables
env = None
# LOG_DIR = None
TRAIN_DIR = None
# BASED_DIR = None
READ_NETWORK_DIR = None
# SAVE_NETWORK_DIR = None
# FILE_SUCCESS = None
# FILE_REWARD = None
# FILE_STEP = None
ACTION_NORM = None

# specify the version of test model
VERSION = "virf_change_action_10"

# used in pre-process the picture
RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128

# parameters used in testing
# their settings relates to other files
ACTIONS = 5 # number of valid actions
PAST_FRAME = 3

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

W_fc_info = weight_variable([PAST_FRAME, 64])
b_fc_info = bias_variable([64])

# input layer
# one state to train each time
s = tf.placeholder(dtype=tf.float32, name='s', shape=(None, RESIZE_WIDTH, RESIZE_HEIGHT, PAST_FRAME))
past_info = tf.placeholder(dtype=tf.float32, name='past_info', shape=(None, PAST_FRAME))
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
# h_drop_fc1 = tf.nn.dropout(h_fc1, keep_prob=0.5)
h_bn_fc1 = tf.layers.batch_normalization(h_fc1, axis=-1, training=training, momentum=0.9)
h_relu_fc1 = tf.nn.relu(h_bn_fc1) # [None, 256]
    
h_fc2 = tf.matmul(h_relu_fc1, W_fc2) + b_fc2
# h_drop_fc2 = tf.nn.dropout(h_fc2, keep_prob=0.5)
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
    # init the virtual test environment
    test_env = env.FocusEnv([FLAGS.TEST_PATH, FLAGS.MAX_STEPS, FLAGS.MIN_ANGLE, FLAGS.MAX_ANGLE])
    # init variables
    success_rate = 0.0
    step_cost = 0.0
    stepList = [] # to record the distribution
    '''
    Start tensorflow
    '''
    # saving and loading networks
    saver = tf.train.Saver()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.PER_GPU_USAGE)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # load in half-trained networks
        checkpoint = tf.train.get_checkpoint_state(FLAGS.READ_NETWORK_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print("Could not find old network weights")

        # start test
		for test in range(FLAGS.TEST_ROUND):
		    init_angle, init_img_path = test_env.reset()
		                
		    # generate the first state, a_past is 0
		    img_t = cv2.imread(init_img_path)
		    img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
		    s_t = np.stack((img_t, img_t, img_t) , axis=2)
                    action_t = np.stack((0.0, 0.0, 0.0), axis=0)
		    past_info_t = action_t
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
			a_input = test_env.actions[action_index]
			# run the selected action and observe next state and reward
			angle_new, img_path_t1, terminal, success = test_env[l].test_step(a_input)

			if terminal:
				# save_last_pic(test, test_env.cur_state, test_env.dic[test_env.cur_state])
			        success_cnt += int(success) # only represents the rate of active terminate
			        total_steps += step
                    stepList += step
			        break
			            
			img_t1 = cv2.imread(img_path_t1)
			img_t1 = cv2.cvtColor(cv2.resize(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
			img_t1 = np.reshape(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT, 1)) # reshape, ready for insert
			action_new = np.reshape(a_input/ACTION_NORM, (1,))
			s_t1 = np.append(img_t1, s_t[:, :, :PAST_FRAME-1], axis=2)
			action_t1 = np.append(action_new, action_t[:PAST_FRAME-1], axis=0)
		        past_info_t1 = action_t1
			# print test info
			print("TEST EPISODE", test, "/ TIMESTEP", step, "/ GRP", test_env.train_data_dir, \
				"/ CURRENT ANGLE", test_env.cur_state, "/ ACTION", a_input)

			# update
			s_t = s_t1
			action_t = action_t1
            		past_info_t = action_t
			step += 1

    	success_rate = success_cnt / FLAGS.TEST_ROUND
    	step_cost = total_steps / FLAGS.TEST_ROUND
    
    print("test grp:", FLAGS.TEST_PATH, "success_rate:", success_rate, "step per episode:", step_cost)
    plot_result()
    return success_rate

'''
save_terminal_pic - save the final picture and use the episode num and
		    the final angle to name it
'''
def save_terminal_pic(epi_num, angle_new, img_path_t1):
    img = cv2.imread(img_path_t1)
    # to avoid '.' appears in file name
    new_pic_name = str(epi_num) + '_' + str(angle_new).replace(".", "_", 1)
    new_pic_path = os.path.join(TEST_DIR, new_pic_name)
    cv2.imwrite(new_pic_path, img)

'''
main
'''
def main(_):
    global TRAIN_DIR, READ_NETWORK_DIR, ACTION_NORM, env
    # import env
    env = __import__(FLAGS.ENV_PATH)
    # normalize the action
    ACTION_NORM = 0.3*env.TIMES

    # directories in training
    TRAIN_DIR = PATH + "/training/" + FLAGS.VERSION
    # the following files are all in training directories
    READ_NETWORK_DIR = TRAIN_DIR + "/saved_networks_" + FLAGS.VERSION
    # start testing!
    testNetwork()

if __name__ == '__main__':
	tf.app.run()
