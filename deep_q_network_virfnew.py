####################################################################################
# This file is the dqn reinforcement learning.
# Modified by xfyu on Apr 9
# Can use "tensorboard --logdir /tmp/logdir" to check current state on
# "localhost:6006".
# 
# Environment: Tensorflow 1.6.0 GPU
#              /usr/local/lib/python2.7/dist-packages/tensorflow
# Tensorboard: change the port and start tensorboard:
#              tensorboard --host=162.105.93.130 --port=6099 --logdir="/home/robot/RL/trainlog"
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
from collections import deque
# import pycontrol as ur
# import trainenv_fnoaction_v3 as env
from importlib import import_module # for dynamic import env
from ctypes import *
import matplotlib.pyplot as plt
import time

###################################################################################
# Important global parameters
###################################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]

# tf.app.flags defined input parameters
# Necessary: VERSION, ENV_PATH
tf.app.flags.DEFINE_string('IMAGE_PATH', '/home/robot/RL/data/new_grp2','train image path')
tf.app.flags.DEFINE_string('TEST_PATH', '/home/robot/RL/data/new_grp2','test image path')
tf.app.flags.DEFINE_string('VERSION', 'virf_grp2_changepoint10', 'version of this training')
tf.app.flags.DEFINE_string('BASED_VERSION', '', 'version of the based model')
tf.app.flags.DEFINE_string('ENV_PATH', 'trainenv_virf_v5', 'path of environment class file')
tf.app.flags.DEFINE_integer('NUM_TRAINING_STEPS', 100000, 'number of time steps in one training')
tf.app.flags.DEFINE_integer('OBSERVE', 1000, 'number of time steps to observe before training')
tf.app.flags.DEFINE_integer('EXPLORE', 50000, 'number of time steps to explore after observation')
tf.app.flags.DEFINE_integer('REPLAY_MEMORY', 500, 'number of previous transitions to remember')
tf.app.flags.DEFINE_float('LEARNING_RATE', 0.001, 'learning rate for optimizer')
tf.app.flags.DEFINE_integer('TEST_ROUND', 50, 'how many episodes in the test')
tf.app.flags.DEFINE_float('GAMMA', 0.99, 'decay rate of past observations')
tf.app.flags.DEFINE_integer('BATCH', 32, 'size of minibatch')
tf.app.flags.DEFINE_float('FINAL_EPSILON', 0.001, 'final value of epsilon')
tf.app.flags.DEFINE_float('INITIAL_EPSILON', 0.01, 'starting value of epsilon')
tf.app.flags.DEFINE_integer('COST_RECORD_STEP', 100, 'cost recording step')
tf.app.flags.DEFINE_integer('NETWORK_RECORD_STEP', 1000, 'network recording step')
tf.app.flags.DEFINE_integer('REWARD_RECORD_STEP', 100, 'reward recording step')
tf.app.flags.DEFINE_integer('STEP_RECORD_STEP', 100, 'step recording step')
tf.app.flags.DEFINE_integer('SUCCESS_RATE_TEST_STEP', 1000, 'testing accuracy step')
tf.app.flags.DEFINE_float('PER_GPU_USAGE', 0.6, 'how much space taken per gpu')
tf.app.flags.DEFINE_string('GPU_LIST', '0, 1', 'how much space taken per gpu')
tf.app.flags.DEFINE_integer('MAX_STEPS', 10, 'max steps defined in env')
tf.app.flags.DEFINE_float('MIN_ANGLE', 30.0, 'min angle defined in env')
tf.app.flags.DEFINE_float('MAX_ANGLE', 69.0, 'max angle defined in env')
FLAGS = tf.app.flags.FLAGS

# define global variables
env = None
LOG_DIR = None
TRAIN_DIR = None
BASED_DIR = None
READ_NETWORK_DIR = None
SAVE_NETWORK_DIR = None
FILE_SUCCESS = None
FILE_REWARD = None
FILE_STEP = None
ACTION_NORM = None

# define fixed parameters
# used in pre-process the picture
RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128

# parameters used in training but not set by flag
# their settings relates to other files
ACTIONS = 5 # number of valid actions
PAST_FRAME = 3 # how many frame in one state
# This file is the dqn reinforcement learning.

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
Neural Network Definitions
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
    optimizer = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(cost)

'''
trainNetwork - the training process
'''
def trainNetwork():	
    '''
    Training Preparations
    '''
    # store the previous observations in replay memory
    D = deque()

    # init the success rate
    success_rate = 0.0

    # initialize several different environment
    train_env = env.FocusEnv([FLAGS.IMAGE_PATH, FLAGS.MAX_STEPS, FLAGS.MIN_ANGLE, FLAGS.MAX_ANGLE]) # init an environment

    '''
    Start tensorflow
    '''
    # saving and loading networks
    saver = tf.train.Saver()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.PER_GPU_USAGE)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # define a summary operation to gather all scalar record
        merged_summary_op = tf.summary.merge_all()
        # define the writer and the directory for it
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        # layout the dashboard
        layout_dashboard(train_writer)
	
	if FLAGS.BASED_VERSION: # check whether there's a based version
        	# load in half-trained networks
        	checkpoint = tf.train.get_checkpoint_state(READ_NETWORK_DIR)
        	if checkpoint and checkpoint.model_checkpoint_path:
            		saver.restore(sess, checkpoint.model_checkpoint_path)
            		print("Successfully loaded:", checkpoint.model_checkpoint_path)
        	else:
            		print("Could not find old network weights")
    
        # rList = []
        # stepList = []
        epsilon = FLAGS.INITIAL_EPSILON # may change with t
        t = 0 # total training steps count
        i = 0 # num of episodes

        # This file is the dqn reinforcement learning.
        # start
        while t <= FLAGS.NUM_TRAINING_STEPS:
            init_angle, init_img_path = train_env.reset()
            rAll = 0 # total reward clear
            step = 0 # stpes in one episode
                
            # generate the first state, a_past is 0
            print(init_angle, init_img_path)
            img_t = cv2.imread(init_img_path)
            img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
            s_t = np.stack((img_t, img_t, img_t) , axis=2)
            action_t = np.stack((0.0, 0.0, 0.0), axis=0)
	    past_info_t = action_t

            # start one episode
            while True:
                # readout_t = readout.eval(feed_dict={s:[s_t], action:[action_t]})[0]
                readout_t, h_pool4_flat_t, h_relu_fc1_t, h_relu_fc2_t = sess.run([readout, h_pool4_flat, h_relu_fc1, h_relu_fc2], feed_dict={
			s : [s_t], 
			past_info : [past_info_t],
			training : False}
		)
                readout_t = readout_t[0]
			
		# print(h_pool4_flat_t)
	        # print(h_relu_fc1_t)
                # print(h_relu_fc2_t)
                print(past_info_t)
                print(readout_t)                
 
               	action_index = 0
                # epsilon-greedy
                if random.random() <= epsilon:
                    print("----------Random Action-----------")
                    action_index = random.randrange(ACTIONS) 
                else:
            	    action_index = np.argmax(readout_t)
            	# finish if
            	a_input = train_env.actions[action_index]
            	a_t = np.zeros([ACTIONS])
            	a_t[action_index] = 1
            
            	# scale down epsilon
            	if epsilon > FLAGS.FINAL_EPSILON and t > FLAGS.OBSERVE:
            	    epsilon -= (FLAGS.INITIAL_EPSILON - FLAGS.FINAL_EPSILON) / FLAGS.EXPLORE
                
            	# run the selected action and observe next state and reward
            	angle_new, img_path_t1, r_t, terminal = train_env.step(a_input)

                # for debug
                # print(angle_new, img_path_t1)
            
            	img_t1 = cv2.imread(img_path_t1)
            	img_t1 = cv2.cvtColor(cv2.resize(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
                img_t1 = np.reshape(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT, 1)) # reshape, ready for insert
                action_new = np.reshape(a_input/ACTION_NORM, (1,))
            	# stack to the state information
		s_t1 = np.append(img_t1, s_t[:, :, :PAST_FRAME-1], axis=2)
                action_t1 = np.append(action_new, action_t[:PAST_FRAME-1], axis=0)
		past_info_t1 = action_t1                   	
		print(past_info_t1) 
            	# store the transition into D
            	D.append((s_t, past_info_t, a_t, r_t, s_t1, past_info_t1, terminal))
            	if len(D) > FLAGS.REPLAY_MEMORY:
           	    	D.popleft()

                '''
                Training
                '''
            	# only train if done observing
            	if t > FLAGS.OBSERVE:
            	    # sample a minibatch to train on
            	    minibatch = random.sample(D, FLAGS.BATCH)

            	    # get the batch variables
            	    s_j_batch = [d[0] for d in minibatch]
                    past_info_j_batch = [d[1] for d in minibatch]
            	    a_batch = [d[2] for d in minibatch]
            	    r_batch = [d[3] for d in minibatch]
            	    s_j1_batch = [d[4] for d in minibatch]
                    past_info_j1_batch = [d[5] for d in minibatch]

            	    y_batch = [] # y is TD target
            	    readout_j1_batch = readout.eval(feed_dict = {
			s : s_j1_batch, 
			past_info : past_info_j1_batch,
			training : False}
		    )
		    # calculate y_batch
            	    for k in range(len(minibatch)):
                	terminal_sample = minibatch[k][6]
                	# if terminal, only equals reward
                	if terminal_sample:
                	    	y_batch.append(r_batch[k])
                	else:
                	    	y_batch.append(r_batch[k] + FLAGS.GAMMA * np.max(readout_j1_batch[k]))
                    # end for
                                
            	    # perform gradient step and record
            	    if t % FLAGS.COST_RECORD_STEP == 0:
                	summary_str, _ = sess.run([merged_summary_op, train_step], feed_dict = {
                		y : y_batch,
                		a : a_batch,
                		s : s_j_batch,
				past_info : past_info_j_batch,
				training : True,
                                accuracy : success_rate}
                	)
                	train_writer.add_summary(summary_str, t) # write cost to record
            	    else:
                	train_step.run(feed_dict = {
                		y : y_batch,
                		a : a_batch,
                		s : s_j_batch,
				past_info : past_info_j_batch,
				training : True}
                	)

                # print info
                state = ""
                if t <= FLAGS.OBSERVE:
                        state = "observe"
                elif t > FLAGS.OBSERVE and t <= FLAGS.OBSERVE + FLAGS.EXPLORE:
                        state = "explore"
                else:
                        state = "train" 
           
                print("EPISODE", i, "/ TIMESTEP", t, "/ GRP", train_env.train_data_dir, "/ STEP", step, "/ STATE", state, \
                	"/ EPSILON", epsilon, "/ CURRENT ANGLE", train_env.cur_state, \
                        "/ ACTION", a_input, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
                    
            	# save progress
            	if t % FLAGS.NETWORK_RECORD_STEP == 0:
            	        saver.save(sess, SAVE_NETWORK_DIR+'/dqn', global_step = t)

                '''
                Testing
                '''
                if (t+1) % FLAGS.SUCCESS_RATE_TEST_STEP == 0:
                        success_rate = testNetwork()
			write_success_rate(t, success_rate)
			
		# update the old values
		s_t = s_t1
		action_t = action_t1
		past_info_t = action_t			
		t += 1
		rAll += r_t
	        step += 1

                if terminal:    
                    	break
                # end one episode, while True

            print("TOTAL REWARD:", rAll)
            # record total reward and step in this episode
            write_reward_and_step(i, rAll, step)
            	
            i += 1 # update num of episodes
        
        train_writer.close()
	sess.close()

    plot_data()
    return

'''
testNetwork - test the training performance, calculate the success rate

Input: s, action,readout
Return: success rate
'''
def testNetwork():
    # initialize testing environment
    test_env = env.FocusEnv([FLAGS.TEST_PATH, FLAGS.MAX_STEPS, FLAGS.MIN_ANGLE, FLAGS.MAX_ANGLE])
    success_cnt = 0.0

    '''
    train set
    '''
    for test in range(FLAGS.TEST_ROUND):
        init_angle, init_img_path = test_env.reset()
                
        # generate the first state, a_past is 0
        img_t = cv2.imread(init_img_path)
        img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
        s_t = np.stack((img_t, img_t, img_t) , axis=2)
        action_t = np.stack((0.0, 0.0, 0.0), axis=0)
	past_info_t = action_t
	step = 0
        # start 1 episode
        while True:
	        # run the network forwardly
        	readout_t = readout.eval(feed_dict={
			s : [s_t], 
			past_info : [past_info_t],
			training : False})[0]
	    	# print(readout_t)
            	# determine the next action
            	action_index = np.argmax(readout_t)
            	a_input = test_env.actions[action_index]
            	# run the selected action and observe next state and reward
            	angle_new, img_path_t1, terminal, success = test_env.test_step(a_input)

            	if terminal:
                	success_cnt += int(success)
                	break
            
            	img_t1 = cv2.imread(img_path_t1)
            	img_t1 = cv2.cvtColor(cv2.resize(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
            	img_t1 = np.reshape(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT, 1)) # reshape, ready for insert
            	action_new = np.reshape(a_input/ACTION_NORM, (1,))
            	s_t1 = np.append(img_t1, s_t[:, :, :PAST_FRAME-1], axis=2)
            	action_t1 = np.append(action_new, action_t[:PAST_FRAME-1], axis=0)
            	# print test info
            	print("TEST EPISODE", test, "/ TIMESTEP", step, "/ GRP", test_env.train_data_dir, \
                	"/ CURRENT ANGLE", test_env.cur_state, "/ ACTION", a_input)

            	# update
            	s_t = s_t1
            	action_t = action_t1
		past_info_t = action_t
            	step += 1

    success_rate = success_cnt / FLAGS.TEST_ROUND
    print("success rate", success_rate)
    return success_rate

'''
write_success_rate - write test result to txt file

Note: If it's the first time record(t = 0), need to erase the past data completely.
'''
def write_success_rate(t, success_rate):
    if t == FLAGS.SUCCESS_RATE_TEST_STEP-1:
        with open(FILE_SUCCESS, 'w') as f:
            txtData = str(success_rate) + '\n'
            f.write(txtData)
    else:
        with open(FILE_SUCCESS, 'a+') as f:
            txtData = str(success_rate) + '\n'
            f.write(txtData)
    return

'''
write_reward_and_step - write those two information in one episode to txt file

Note: if it's the first episode(i = 0), need to erase the past data completely.
'''
def write_reward_and_step(i, rAll, step):
    # finish one episode, record this step
    if i == 0: # first time
        with open(FILE_REWARD, 'w') as f:
            txtData = str(rAll) + '\n'
            f.write(txtData)
        with open(FILE_STEP, 'w') as f:
            txtData = str(step) + '\n'
            f.write(txtData)
        return
    if i % FLAGS.REWARD_RECORD_STEP == 0:
        with open(FILE_REWARD, 'a+') as f:
            txtData = str(rAll) + '\n'
            f.write(txtData)
    if i % FLAGS.STEP_RECORD_STEP == 0:
        with open(FILE_STEP, 'a+') as f:
            txtData = str(step) + '\n'
            f.write(txtData)
    return

'''
plot_reward - plot rList and stepList

Input: rList - the record of reward changing
       stepList - the record of steps
'''
def plot_data():
    rList = []
    stepList = []
    successList = []
    with open(FILE_REWARD, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rList.append(float(line))
    with open(FILE_STEP, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stepList.append(float(line))
    with open(FILE_SUCCESS, 'r') as f:
        lines = f.readlines()
        for line in lines:
            successList.append(float(line))
    plt.figure()
    # plot rList
    plt.subplot(221)
    plt.plot(rList, 'b')
    plt.xlabel('episode({})'.format(FLAGS.REWARD_RECORD_STEP))
    plt.ylabel('reward')

    # plot stepList
    plt.subplot(222)
    plt.plot(stepList, 'r')
    plt.xlabel('episode({})'.format(FLAGS.STEP_RECORD_STEP))
    plt.ylabel('steps')

    # plot successList
    plt.subplot(223)
    plt.plot(successList, 'g')
    plt.xlabel('episode({})'.format(FLAGS.SUCCESS_RATE_TEST_STEP))
    plt.ylabel('train accuracy')

    # save this figure
    plt.savefig(TRAIN_DIR + '/result_' + str(FLAGS.VERSION), dpi=600)
    return

'''
layout_dashboard - call once to init the dashboard
                   or nothing displays on the website
'''
def layout_dashboard(writer):
    from tensorboard import summary
    from tensorboard.plugins.custom_scalar import layout_pb2
    
    # This action does not have to be performed at every step, so the action is not
    # taken care of by an op in the graph. We only need to specify the layout once. 
    # We only need to specify the layout once (instead of per step).
    layout_summary = summary.custom_scalar_pb(layout_pb2.Layout(
        category=[
            layout_pb2.Category(
            title='losses',
            chart=[
                layout_pb2.Chart(
                    title='losses',
                    multiline=layout_pb2.MultilineChartContent(
                    tag=[r'loss.*'],
                )),
                layout_pb2.Chart(
                    title='baz',
                    margin=layout_pb2.MarginChartContent(
                    series=[
                        layout_pb2.MarginChartContent.Series(
                        value='loss/baz/scalar_summary',
                        lower='baz_lower/baz/scalar_summary',
                        upper='baz_upper/baz/scalar_summary'),
                    ],
                )), 
            ]),
            layout_pb2.Category(
            title='trig functions',
            chart=[
                layout_pb2.Chart(
                    title='wave trig functions',
                    multiline=layout_pb2.MultilineChartContent(
                    tag=[r'trigFunctions/cosine', r'trigFunctions/sine'],
                )),
                # The range of tangent is different. Let's give it its own chart.
                layout_pb2.Chart(
                    title='tan',
                    multiline=layout_pb2.MultilineChartContent(
                    tag=[r'trigFunctions/tangent'],
                )),
            ],
        # This category we care less about. Let's make it initially closed.
        closed=True),
    ]))
    writer.add_summary(layout_summary)

###################################################################################
# Main
###################################################################################
def main(_): # must have input parameter
	global LOG_DIR, TRAIN_DIR, BASED_DIR, READ_NETWORK_DIR, SAVE_NETWORK_DIR
	global FILE_SUCCESS, FILE_REWARD, FILE_STEP, ACTION_NORM, env
	# import env
	env = __import__(FLAGS.ENV_PATH)
    	# normalize the action
    	ACTION_NORM = 0.3*env.TIMES

	# define variables
	LOG_DIR = PATH + "/virlog/" + FLAGS.VERSION
	TRAIN_DIR = PATH + "/virtraining/" + FLAGS.VERSION
	BASED_DIR = PATH + "/virtraining/" + FLAGS.BASED_VERSION
	# if directory does not exist, new it
	if not os.path.isdir(TRAIN_DIR):
	        os.makedirs(TRAIN_DIR)
	# the following files are all in training directories
	READ_NETWORK_DIR = BASED_DIR + "/saved_networks_" + FLAGS.BASED_VERSION
	SAVE_NETWORK_DIR = TRAIN_DIR + "/saved_networks_" + FLAGS.VERSION
	# saved networks are in train directory of specified version
	if not os.path.isdir(SAVE_NETWORK_DIR):
		os.makedirs(SAVE_NETWORK_DIR)
	FILE_SUCCESS = TRAIN_DIR + "/success_rate_" + FLAGS.VERSION + ".txt"
	FILE_REWARD = TRAIN_DIR + "/total_reward_" + FLAGS.VERSION + ".txt"
	FILE_STEP = TRAIN_DIR + "/step_cnt_" + FLAGS.VERSION + ".txt"

    	# set GPU
    	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_LIST
	
	# start training!
	trainNetwork()

if __name__ == "__main__":
	tf.app.run() # execute main
