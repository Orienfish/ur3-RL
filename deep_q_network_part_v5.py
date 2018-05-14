####################################################################################
# This file is the dqn reinforcement learning.
# Modified by xfyu on Apr 9
# Can use "tensorboard --logdir /tmp/logdir" to check current state on
# "localhost:6006".
# 
# Environment: Tensorflow 1.6.0 GPU
#              /usr/local/lib/python2.7/dist-packages/tensorflow
# Tensorboard: change the port and start tensorboard:
#              tensorboard --host=162.105.93.130 --port=6099 --logdir="/tmp/logdir"
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
import trainenv_aa_part_v4 as env
from ctypes import *
import matplotlib.pyplot as plt
import time

###################################################################################
# Important global parameters
###################################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
IMAGE_PATH = ['/home/robot/RL/grp1_part/']# ,'/home/robot/RL/grp2/','/home/robot/RL/grp3/',\
#   '/home/robot/RL/grp4/', '/home/robot/RL/grp5/']
TEST_PATH = '/home/robot/RL/grp1_part/'
DICT_PATH = 'dict.txt'
ANGLE_LIMIT_PATH = 'angle.txt'
VERSION = "v5"
LOG_DIR = "/tmp/logdir/train_part_" + VERSION
READ_NETWORK_DIR = "saved_networks" # not use, from scratch
SAVE_NETWORK_DIR = "saved_networks_part_" + VERSION
# if directory does not exist, new it
if not os.path.isdir(os.path.join(PATH, SAVE_NETWORK_DIR)):
	os.makedirs(os.path.join(PATH, SAVE_NETWORK_DIR))
FILE_SUCCESS = "success_rate_" + VERSION + ".txt"
FILE_REWARD = "total_reward_" + VERSION + ".txt"
FILE_STEP = "step_cnt_" + VERSION + ".txt"
# used in pre-process the picture
RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128
# normalize the action
ACTION_NORM = 2.7
ANGLE_NORM = 100

# parameters used in training
ACTIONS = 5 # number of valid actions
GAMMA = 0.99 # in DQN. decay rate of past observations
PAST_FRAME = 3 # how many frame in one state
LEARNING_RATE = 0.0001 # parameter in the optimizer
NUM_TRAINING_STEPS = 50000 # times of episodes in one folder
REPLAY_MEMORY = 500 # number of previous transitions to remember
BATCH = 32 # size of minibatch
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 40000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.01 # starting value of epsilon
COST_RECORD_STEP = 100
NETWORK_RECORD_STEP = 100
REWARD_RECORD_STEP = 100
STEP_RECORD_STEP = 100
SUCCESS_RATE_TEST_STEP = 1000
TEST_ROUND = 20 # how many episodes in the test
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

W_fc_info = weight_variable([PAST_FRAME*2, 64])
b_fc_info = bias_variable([64])

# input layer
# one state to train each time
s = tf.placeholder("float", [None, RESIZE_WIDTH, RESIZE_HEIGHT, PAST_FRAME])
past_info = tf.placeholder("float", [None, PAST_FRAME*2])
training = tf.placeholder_with_default(False, shape=(), name='training')

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
Neural Network Definitions
'''
# define the cost function
a = tf.placeholder("float", [None, ACTIONS])
y = tf.placeholder("float", [None])
# define cost
with tf.name_scope('cost'):
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    tf.summary.scalar('cost', cost)
# define training step
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
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

    # init the environment list
    train_env = []

    '''
    Start tensorflow
    '''
    # saving and loading networks
    saver = tf.train.Saver()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # define a summary operation to gather all scalar record
        merged_summary_op = tf.summary.merge_all()
        # define the writer and the directory for it
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        # layout the dashboard
        layout_dashboard(train_writer)

        # load in half-trained networks
        #checkpoint = tf.train.get_checkpoint_state(READ_NETWORK_DIR)
        #if checkpoint and checkpoint.model_checkpoint_path:
        #    saver.restore(sess, checkpoint.model_checkpoint_path)
        #    print("Successfully loaded:", checkpoint.model_checkpoint_path)
        #else:
        #    print("Could not find old network weights")
    
        # rList = []
        # stepList = []
        epsilon = INITIAL_EPSILON # may change with t
        t = 0 # total training steps count
        i = 0 # num of episodes

        # initialize several different environment
        for p in IMAGE_PATH:
            train_env.append(env.FocusEnv(p+DICT_PATH, p+ANGLE_LIMIT_PATH)) # init an environment
        action_space = train_env[0].actions

        # This file is the dqn reinforcement learning.
        # start
        while t < NUM_TRAINING_STEPS:
            # one episode in each training environment
            for l in range(len(train_env)):
                init_angle, init_img_path = train_env[l].reset()
                rAll = 0 # total reward clear
                step = 0 # stpes in one episode
                
                # generate the first state, a_past is 0
            	img_t = cv2.imread(init_img_path)
            	img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
            	s_t = np.stack((img_t, img_t, img_t), axis=2)
                action_t = np.stack((0.0, 0.0, 0.0), axis=0)
		angle_t = np.stack((init_angle/ANGLE_NORM, init_angle/ANGLE_NORM, init_angle/ANGLE_NORM), axis=0)
		past_info_t = np.append(action_t, angle_t, axis=0)

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
                        print(h_relu_fc2_t)
                        print(readout_t)                
 
               		action_index = 0
                	# epsilon-greedy
                	if random.random() <= epsilon:
                    		print("----------Random Action-----------")
                    		action_index = random.randrange(ACTIONS) 
                	else:
            	    		action_index = np.argmax(readout_t)
            	    	a_input = action_space[action_index]
            		a_t = np.zeros([ACTIONS])
            		a_t[action_index] = 1
            
            		# scale down epsilon
            		if epsilon > FINAL_EPSILON and t > OBSERVE:
            	    		epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
            		# run the selected action and observe next state and reward
            		angle_new, img_path_t1, r_t, terminal = train_env[l].step(a_input)

                	# for debug
                	# print(angle_t1, img_path_t1)
            
            		img_t1 = cv2.imread(img_path_t1)
            		img_t1 = cv2.cvtColor(cv2.resize(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
                	img_t1 = np.reshape(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT, 1)) # reshape, ready for insert
            		angle_new = np.reshape(angle_new/ANGLE_NORM, (1,))
                        action_new = np.reshape(a_input/ACTION_NORM, (1,))
            		# stack to the state information
			s_t1 = np.append(img_t1, s_t[:, :, :PAST_FRAME-1], axis=2)
            		angle_t1 = np.append(angle_new, angle_t[:PAST_FRAME-1], axis=0)
                        action_t1 = np.append(action_new, action_t[:PAST_FRAME-1], axis=0)
			past_info_t1 = np.append(action_t1, angle_t1, axis=0)
                   	print(past_info_t1) 
            		# store the transition into D
            		D.append((s_t, past_info_t, a_t, r_t, s_t1, past_info_t1, terminal))
            		if len(D) > REPLAY_MEMORY:
           	    		D.popleft()

                        '''
                        Training
                        '''
            		# only train if done observing
            		if t > OBSERVE:
            	    		# sample a minibatch to train on
            	    		minibatch = random.sample(D, BATCH)

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
            	    		for k in range(len(minibatch)):
                			terminal_sample = minibatch[k][6]
                			# if terminal, only equals reward
                			if terminal_sample:
                	    			y_batch.append(r_batch[k])
                			else:
                	    			y_batch.append(r_batch[k] + GAMMA * np.max(readout_j1_batch[k]))
                                
            	    		# perform gradient step and record
            	    		if t % COST_RECORD_STEP == 0:
                			summary_str, _ = sess.run([merged_summary_op, train_step], feed_dict = {
                				y : y_batch,
                				a : a_batch,
                				s : s_j_batch,
						past_info : past_info_j_batch,
						training : True}
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
                    	if t <= OBSERVE:
                        	state = "observe"
                    	elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                        	state = "explore"
                    	else:
                        	state = "train" 
           
                    	print("EPISODE", i, "/ TIMESTEP", t, "/ GRP", train_env[l].dict_path, "/ STEP", step, "/ STATE", state, \
                        	"/ EPSILON", epsilon, "/ CURRENT ANGLE", train_env[l].cur_state, \
                            	"/ ACTION", a_input, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
                    
            		# update the old values
            		s_t = s_t1
                	angle_t = angle_t1
			action_t = action_t1
            		t += 1    # total time steps
                	rAll += r_t
                        step += 1
    
            		# save progress
            		if t % NETWORK_RECORD_STEP == 0:
            	    		saver.save(sess, SAVE_NETWORK_DIR+'/dqn', global_step = t)
    
            		
                	# time.sleep(1)

                        '''
                        Testing
                        '''
                        if t % SUCCESS_RATE_TEST_STEP == 0:
                        	success_rate = testNetwork()
                        	write_success_rate(t, success_rate)

                	if terminal:    
                    		break

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
    test_env = env.FocusEnv(TEST_PATH+DICT_PATH, TEST_PATH+ANGLE_LIMIT_PATH)
    action_space = test_env.actions
    success_cnt = 0.0
    for test in range(TEST_ROUND):
        init_angle, init_img_path = test_env.reset()
                
        # generate the first state, a_past is 0
        img_t = cv2.imread(init_img_path)
        img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
        s_t = np.stack((img_t, img_t, img_t), axis=2)
        angle_t = np.stack((init_angle/ANGLE_NORM, init_angle/ANGLE_NORM, init_angle/ANGLE_NORM), axis=0)
        action_t = np.stack((0.0, 0.0, 0.0), axis=0)
	past_info_t = np.append(action_t, angle_t, axis=0)
        step = 0
        # start 1 episode
        while True:
            # run the network forwardly
            readout_t = readout.eval(feed_dict={
		s : [s_t], 
		past_info : [past_info_t],
		training : False})[0]
	    print(readout_t)
            # determine the next action
            action_index = np.argmax(readout_t)
            a_input = action_space[action_index]
            # run the selected action and observe next state and reward
            angle_new, img_path_t1, terminal, success = test_env.test_step(a_input)

            if terminal:
                success_cnt += int(success)
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
            print("TEST EPISODE", test, "/ TIMESTEP", step, "/ GRP", test_env.dict_path, \
                "/ CURRENT ANGLE", test_env.cur_state, "/ ACTION", a_input)

            # update
            s_t = s_t1
            action_t = action_t1
	    angle_t = angle_t1
            step += 1

    success_rate = success_cnt / TEST_ROUND
    print("success_rate:", success_rate)
    return success_rate

'''
write_success_rate - write test result to txt file

Note: If it's the first time record(t = 0), need to erase the past data completely.
'''
def write_success_rate(t, success_rate):
    if t == 0:
        with open(FILE_SUCCESS, 'w') as f:
            txtData = str(success_rate) +'\n'
            f.write(txtData)
    else:
        with open(FILE_SUCCESS, 'a+') as f:
            txtData = str(success_rate) +'\n'
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
    if i % REWARD_RECORD_STEP == 0:
        with open(FILE_REWARD, 'a+') as f:
            txtData = str(rAll) + '\n'
            f.write(txtData)
    if i % STEP_RECORD_STEP == 0:
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
    file
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
    plt.plot(rList, 'b.-')
    plt.xlabel('episode({})'.format(REWARD_RECORD_STEP))
    plt.ylabel('reward')

    # plot stepList
    plt.subplot(222)
    plt.plot(stepList, 'r*-')
    plt.xlabel('episode({})'.format(STEP_RECORD_STEP))
    plt.ylabel('steps')

    # plot stepList
    plt.subplot(212)
    plt.plot(successList, 'go-')
    plt.xlabel('episode({})'.format(SUCCESS_RATE_TEST_STEP))
    plt.ylabel('steps')

    # save this figure
    plt.savefig('result', dpi=1200)
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
if __name__ == "__main__":
	trainNetwork()
