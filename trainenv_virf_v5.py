##################################################################
# This file is the virtual training environment.TER->DO_NOTHING
# Include the state transfer. All the image processing are not here.
# Modified by xfyu on April 25, 2018.
##################################################################
# !/usr/bin/env python

import os
import random
import time
import matplotlib.pyplot as plt
import cv2
import numpy

#################################################################
# Important global parameters
#################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
# IMAGE_PATH = '/home/robot/RL/grp1'
SUCCESS_REWARD = 100
FAILURE_REWARD = -100
FOCUS_REWARD_NORM = 10.0
# ACTION_REWARD = 1

# properties that are set from deep_q_network
# MAX_STEPS = 10 # set by parameters from deep_q_network
# MAX_ANGLE = 69.0
# MIN_ANGLE = 30.0

# judge standart
CHANGE_POINT_THRES = 0.6
SUCCESS_THRES = 0.9

# VERY IMPORTANT!!!
TIMES = 9
# actions
COARSE_POS = 0.3 * TIMES
FINE_POS = 0.3
TERMINAL = 0
FINE_NEG = -0.3
COARSE_NEG = -0.3 * TIMES

RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128


class FocusEnv():  # one class for one folder
    def __init__(self, info):  # info:[TRAIN_DATA_DIR, MAX_STEPS, MIN_ANGLE, MAX_ANGLE]
        # define the action space
        self.actions = [COARSE_NEG, FINE_NEG, TERMINAL, FINE_POS, COARSE_POS]
        self.train_data_dir = info[0]
        self.max_steps, self.min_angle, self.max_angle = info[1], info[2], info[3]

        self.get_max_focus()  # set focus points

    # self.angle_list_path = os.path.join(self.train_data_dir, "angleList.txt")
    # self.import_angle()

    def reset(self):  # reset starts a new episode, never change DICT_PATH and ANGLE_LIMIT_PATH
        # reset the starting state
        total_state_cnt = int((self.max_angle - self.min_angle) / 0.3 + 1)
        index = int(random.random() * total_state_cnt)  # randomly initialized
        # start from 0
        self.cur_state = self.min_angle + index * 0.3  # self.dic.keys()[index]
        self.cur_state = round(self.cur_state, 2)
        self.cur_step = 0
        init_path = os.path.join(self.train_data_dir, str(self.cur_state) + '.jpg')
        # define the last focus
        # img = cv2.imread(init_path)
        # img = cv2.cvtColor(cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
        # self.last_focus = TENG(img)
        # return the angle of first state and the image name
        print(init_path)
        return self.cur_state, init_path

    '''
    step - regulations of transfering between states

    Input: input_action
    	   time_step - for updating the terminal range
    	   EXPLORE - for updating the terminal range
    Return: next_state - the total angle of next state
    		next_image_path - the image path of next state(the dictionary is not accessible from outside)
    		reward - reward of this single operation
    		terminal - True or False
    '''

    def step(self, input_action):  # action is the angle to move
        self.cur_step = self.cur_step + 1
        next_state = self.cur_state + input_action
        next_state = round(next_state, 2)
        next_image_path = os.path.join(self.train_data_dir, str(next_state) + '.jpg')
        self.cur_state = next_state  # state transfer
        # calculate the focus
        pic = cv2.imread(next_image_path)
        pic = cv2.cvtColor(cv2.resize(pic, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
        focus = TENG(pic)

        # special termination
        if self.cur_state > self.max_angle or self.cur_state < self.min_angle:
            return self.cur_state, next_image_path, FAILURE_REWARD + self.get_reward(focus), True

        # choose to terminate
        if input_action == TERMINAL:
            if focus > self.success_focus:
                return self.cur_state, next_image_path, SUCCESS_REWARD, True
            return self.cur_state, next_image_path, FAILURE_REWARD + self.get_reward(focus), True

        # special case - failure
        if self.cur_step >= self.max_steps:
            return self.cur_state, next_image_path, FAILURE_REWARD + self.get_reward(focus), True

        '''
	action_reward = -ACTION_REWARD # determine the action reward
	if focus > self.change_point_focus:
		if abs(input_action) < 1.0: # fine tune
			action_reward = ACTION_REWARD	
	else:
		if abs(input_action) > 1.0: # coarse tune
			action_reward = ACTION_REWARD
	'''
        # return self.cur_state, next_image_path, action_reward + self.get_reward(focus), False
        return self.cur_state, next_image_path, self.get_reward(focus), False

    '''
    get_reward - reward determination
    generate reward from the current taking pictures, should be less than 0
    '''

    def get_reward(self, cur_focus):
        reward = (cur_focus - self.success_focus) * FOCUS_REWARD_NORM / self.max_focus
        print("max", self.success_focus, "cur", cur_focus, "reward is", reward)
        # self.last_focus = cur_focus # update
        return reward  # return

    '''
    get_max_focus - get the max focus for reference
    '''

    def get_max_focus(self):
        cur_angle = self.min_angle
        max_focus = 0.0
        # walk through every images in the directory
        while cur_angle <= self.max_angle:
            pic_name = os.path.join(self.train_data_dir, str(cur_angle) + '.jpg')
            print("calculating %s" % pic_name)
            # read and calculate focus
            img = cv2.imread(pic_name)
            img = cv2.cvtColor(cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
            cur_focus = TENG(img)
            if cur_focus > max_focus:
                max_focus = cur_focus
            cur_angle += 0.3  # move to the next picture
        print("max focus is", max_focus)
        self.change_point_focus = max_focus * CHANGE_POINT_THRES
        self.success_focus = max_focus * SUCCESS_THRES
        self.max_focus = max_focus
        return max_focus

    '''
    step - regulations of transfering between states in testing

    Input: input_action
    Return: next_state - the total angle of next state
    	    next_image_path - the image path of next state(the dictionary is not accessible from outside)
    	    terminal - True or False
    	    success - Success or not
    '''

    def test_step(self, input_action):  # action is the angle to move
        self.cur_step = self.cur_step + 1

        # get the next state, no need to calculate reward
        next_state = self.cur_state + input_action
        next_state = round(next_state, 2)
        next_image_path = os.path.join(self.train_data_dir, str(next_state) + '.jpg')
        self.cur_state = next_state  # state transfer
        # calculate focus
        pic = cv2.imread(next_image_path)
        pic = cv2.cvtColor(cv2.resize(pic, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)
        focus = TENG(pic)

        # special termination
        if next_state > self.max_angle or next_state < self.min_angle:
            return next_state, next_image_path, True, False

        # choose to terminate
        if input_action == TERMINAL:
            if focus > self.success_focus:
                return next_state, next_image_path, True, True
            return next_state, next_image_path, True, False

        # special case - failure
        if self.cur_step >= self.max_steps:
            return next_state, next_image_path, True, False

        return next_state, next_image_path, False, False


'''
TENG - Focus calculating function with TENENGRAD
'''


def TENG(img):
    guassianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    guassianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return numpy.mean(guassianX * guassianX +
                      guassianY * guassianY)
