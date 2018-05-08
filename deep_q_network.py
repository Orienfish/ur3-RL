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
import pycontrol as ur
import trainenv as env
from ctypes import *
import matplotlib.pyplot as plt
import time

###################################################################################
# Important global parameters
###################################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
IMAGE_PATH = ['/home/robot/RL/grp1/']# ,'/home/robot/RL/grp2/','/home/robot/RL/grp3/',\
#   '/home/robot/RL/grp4/', '/home/robot/RL/grp5/']
DICT_PATH = 'dict.txt'
ANGLE_LIMIT_PATH = 'angle.txt'
LOG_DIR = "/tmp/logdir/train5"
READ_NETWORK_DIR = "saved_networks" # not use, from scratch
SAVE_NETWORK_DIR = "saved_networks"
# used in pre-process the picture
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 320

# parameters used in training
ACTIONS = 5 # number of valid actions
GAMMA = 0.99 # in DQN. decay rate of past observations
PAST_FRAME = 3 # how many frame in one state
LEARNING_RATE = 0.001 # parameter in the optimizer
NUM_EPISODES = int(1e5) # times of episodes in one folder
REPLAY_MEMORY = 1000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
COST_RECORD_STEP = 100
NETWORK_RECORD_STEP = 1000
REWARD_RECORD_STEP = 1000

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

'''
createNetwork - set the structure of CNN
'''
def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, PAST_FRAME, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([4, 4, 64, 16])
    b_conv3 = bias_variable([16])

    W_fc1 = weight_variable([400, 16])
    b_fc1 = bias_variable([16])

    W_fc2 = weight_variable([16+PAST_FRAME*2+1, 8])
    b_fc2 = bias_variable([8])

    W_fc3 = weight_variable([8, ACTIONS])
    b_fc3 = bias_variable([ACTIONS])

    # input layer
    # one state to train each time
    s = tf.placeholder("float", [None, RESIZE_WIDTH, RESIZE_HEIGHT, PAST_FRAME])
    angle = tf.placeholder("float", [None, PAST_FRAME])
    action = tf.placeholder("float", [None, PAST_FRAME])
    step = tf.placeholder("float", [None, 1])
    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 400])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_add = tf.concat([h_fc1, angle], 1)
    h_fc1_add = tf.concat([h_fc1_add, action], 1)
    h_fc1_add = tf.concat([h_fc1_add, step], 1)
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_add, W_fc2) + b_fc2)
    # readout layer
    readout = tf.matmul(h_fc2, W_fc3) + b_fc3

    return s, angle, action, step, readout # s and past_a are all placeholders 

'''
trainNetwork - the training process
'''
def trainNetwork(s, angle, action, step, readout):
    '''
    Neural Network Definitions
    '''
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    # success rate for display
    success = tf.placeholder("float")
    total = tf.placeholder("float")
    with tf.name_scope('cost'):
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        tf.summary.scalar('cost', cost)
    with tf.name_scope('success_rate'):
        success_rate = tf.divide(success, total)
        tf.summary.scalar('success rate', success_rate)
    # define training step
    with tf.name_scope('train'): 
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    '''
    Training Preparations
    '''
    # store the previous observations in replay memory
    D = deque()

    # init the environment list
    train_env = []

    '''
    Training
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

        # initial several different environment
        for p in IMAGE_PATH:
            train_env.append(env.FocusEnv(p+DICT_PATH, p+ANGLE_LIMIT_PATH)) # init an environment
        action_space = train_env[0].actions

        # start
        while t < NUM_EPISODES:
            # one episode in each training environment
            for l in range(len(train_env)):
                init_angle, init_img_path = train_env[l].reset()
                rAll = 0 # total reward clear
                step = 0
                # generate the first state, a_past is 0
            	img_t = cv2.imread(init_img_path)
            	img_t = cv2.cvtColor(cv2.resize(img_t, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
            	s_t = np.stack((img_t, img_t, img_t), axis=2)
            	angle_t = np.stack((init_angle, init_angle, init_angle), axis=0)
                action_t = np.stack((0.0, 0.0, 0.0), axis=0)
        
            	# start one episode
            	while True:
                	readout_t = readout.eval(feed_dict={s:[s_t], angle:[angle_t]})[0]
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
            		angle_new, img_path_t1, r_t, terminal, cur_step = train_env[l].step(a_input, t, EXPLORE)

                	# for debug
                	# print(angle_t1, img_path_t1)
            
            		img_t1 = cv2.imread(img_path_t1)
            		img_t1 = cv2.cvtColor(cv2.resize(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
                	img_t1 = np.reshape(img_t1, (RESIZE_WIDTH, RESIZE_HEIGHT, 1)) # reshape, ready for insert
            		angle_new = np.reshape(angle_new, (1,))
                        action_new = np.reshape(a_input, (1,))
            		s_t1 = np.append(img_t1, s_t[:, :, :PAST_FRAME-1], axis=2)
            		angle_t1 = np.append(angle_new, angle_t[:PAST_FRAME-1], axis=0)
                        action_t1 = np.append(action_new, action_t[:PAST_FRAME-1], axis=0)
            		# store the transition into D
            		D.append((s_t, angle_t, action_t, a_t, r_t, s_t1, angle_t1, action_t1, terminal, cur_step))
            		if len(D) > REPLAY_MEMORY:
           	    		D.popleft()

            		# only train if done observing
            		if t > OBSERVE:
            	    		# sample a minibatch to train on
            	    		minibatch = random.sample(D, BATCH)

            	    		# get the batch variables
            	    		s_j_batch = [d[0] for d in minibatch]
                    		angle_j_batch = [d[1] for d in minibatch]
                            action_j_batch = [d[2] for d in minibatch] 
            	    		a_batch = [d[3] for d in minibatch]
            	    		r_batch = [d[4] for d in minibatch]
            	    		s_j1_batch = [d[5] for d in minibatch]
            	    		angle_j1_batch = [d[6] for d in minibatch]
                            action_j1_batch = [d[7] for d in minibatch]
                            cur_step_batch = [d[9] for d in minibatch]

            	    		y_batch = [] # y is TD target
            	    		readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch, angle : angle_j1_batch, action : action_j1_batch})
            	    		for k in range(len(minibatch)):
                			terminal_sample = minibatch[k][8]
                			# if terminal, only equals reward
                			if terminal_sample:
                	    			y_batch.append(r_batch[k])
                			else:
                	    			y_batch.append(r_batch[k] + GAMMA * np.max(readout_j1_batch[k]))

                            	# compute the total success rate
                            	success_cnt = 0.0
                            	total_cnt = 0.0
                            	for tmp in range(len(train_env)):
                                	success_cnt += train_env[tmp].success_cnt
                                	total_cnt += train_env[tmp].success_cnt + train_env[tmp].fail_cnt
                                
            	    		# perform gradient step and record
            	    		if t % COST_RECORD_STEP == 0:
                			summary_str, _ = sess.run([merged_summary_op, train_step], feed_dict = {
                				y : y_batch,
                				a : a_batch,
                				s : s_j_batch,
                        			angle : angle_j_batch,
                                    		success : success_cnt,
                                    		total : total_cnt,
                                            action : action_j_batch,
                                            step : cur_step_batch}
                			)
                			train_writer.add_summary(summary_str, t) # write cost to record
            	    		else:
                			train_step.run(feed_dict = {
                				y : y_batch,
                				a : a_batch,
                				s : s_j_batch,
                        			angle: angle_j_batch,
                                    		success : success_cnt,
                                    		total : total_cnt,
                                            action : action_j_batch,
                                            step : cur_step_batch}
                			)

            		# update the old values
            		s_t = s_t1
                	angle_t = angle_t1
			action_t = action_t1
            		t += 1    # total time steps
                	rAll += r_t
           		step += 1 # = steps number in this episode
    
            		# save progress
            		if t % NETWORK_RECORD_STEP == 0:
            	    		saver.save(sess, SAVE_NETWORK_DIR+'/dqn', global_step = t)
    
            		# print info
            		state = ""
            		if t <= OBSERVE:
            			state = "observe"
            		elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            			state = "explore"
            		else:
            			state = "train" 
           
            		print("EPISODE", i, "/ TIMESTEP", t, "/ GRP", train_env[l].dict_path, "/ STATE", state, \
            			"/ EPSILON", epsilon, "/ CURRENT ANGLE", train_env[l].cur_state, \
                    		"/ ACTION", a_input, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
            		print(readout_t)
                	# time.sleep(0.3)

                	if terminal:    
                    		break

            	print("TOTAL REWARD:", rAll)
        
            	# finish one episode, record this step
            	if i == 0: # first time
            		with open("total_reward.txt", 'w') as f:
            			txtData = str(rAll) + '\n'
            			f.write(txtData)
            		with open("step_cnt.txt", 'w') as f:
            			txtData = str(step) + '\n'
            			f.write(txtData)
            	elif i % REWARD_RECORD_STEP == 0:
            		with open("total_reward.txt", 'a+') as f:
                    		txtData = str(rAll) + '\n'
                    		f.write(txtData)
                	with open("step_cnt.txt", 'a+') as f:
                    		txtData = str(step) + '\n'
                    		f.write(txtData)
            	i += 1 # update num of episodes
        
	plot_reward()
        train_writer.close()

'''
plot_reward - plot rList and stepList

Input: rList - the record of reward changing
       stepList - the record of steps
'''
def plot_reward():
    rList = []
    stepList = []
    with open("total_reward.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            rList.append(float(line))
    with open("step_cnt.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            stepList.append(float(line))
    # plot rList
    plt.plot(rList)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig('reward', dpi=1200)
    plt.show()

    # plot stepList
    plt.plot(stepList)
    plt.xlabel('episode')
    plt.ylabel('steps in one episode')
    plt.savefig('step', dpi=1200)
    plt.show()

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

'''
playGame - call createNetwork and train it
'''
def StartTraining():
    s, angle, action, step, readout = createNetwork()
    trainNetwork(s, angle, action, step, readout)

###################################################################################
# Main
###################################################################################
if __name__ == "__main__":
    StartTraining()
