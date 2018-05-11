##################################################################
# This file is the virtual training environment.TER->DO_NOTHING
# Include the state transfer. All the image processing are not here.
# Modified by xfyu on April 25, 2018.
##################################################################
#!/usr/bin/env python

import os
import random
import time
import matplotlib.pyplot as plt
#################################################################
# Important global parameters
#################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
# IMAGE_PATH = '/home/robot/RL/grp1'
SUCCESS_REWARD = 1000
FAILURE_REWARD = -100
MAX_STEPS = 30
# maximum and minimum limitations, a little different from collectenv.py
# only part of the data is used: from 150.jpg to 180.jpg
MAX_ANGLE = 54.0
MIN_ANGLE = 45.0
MIN_ANGLE_LIMIT = 45.3
MAX_ANGLE_LIMIT = 53.7
# actions
COARSE_POS = 0.3*9
FINE_POS = 0.3
DO_NOTHING = 0
FINE_NEG = -0.3
COARSE_NEG = -0.3*9

class FocusEnv(): # one class for one folder
    def __init__(self, DICT_PATH, ANGLE_LIMIT_PATH):
	# define the action space
	self.actions = [COARSE_NEG, FINE_NEG, DO_NOTHING, FINE_POS, COARSE_POS]
	self.dic = dict()
	self.dict_path = DICT_PATH
    	self.angle_path = ANGLE_LIMIT_PATH
	self.import_dic()
	self.import_angle()

    def reset(self): # reset starts a new episode, never change DICT_PATH and ANGLE_LIMIT_PATH
    # reset the starting state
	# index = int(random.random()*len(self.dic.keys())) # randomly initialized
	# start from 0
	self.cur_state = MIN_ANGLE # self.dic.keys()[index] 
	self.cur_step = 0
	# return the angle of first state and the image name
	return self.cur_state, self.dic[self.cur_state]

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
    def step(self, input_action): # action is the angle to move
    	self.cur_step = self.cur_step + 1
    	# special case #1
    	if self.cur_state + input_action > MAX_ANGLE_LIMIT:
    	    return MAX_ANGLE, self.dic[MAX_ANGLE], FAILURE_REWARD + self.get_reward(MAX_ANGLE), True
    	# special case #2
    	if self.cur_state + input_action < MIN_ANGLE_LIMIT:
    	    return MIN_ANGLE, self.dic[MIN_ANGLE], FAILURE_REWARD + self.get_reward(MIN_ANGLE), True
	# special case #3
	if self.cur_step >= MAX_STEPS:
	    next_state = self.cur_state + input_action
	    next_state = round(next_state, 2)
	    next_image_path = self.dic[next_state]
	    # check the final state, set the terminal reward regarding the final angle
	    terminal_reward = FAILURE_REWARD
	    if next_state >= self.terminal_angle_low and next_state <= self.terminal_angle_high:
		terminal_reward = SUCCESS_REWARD
	    return next_state, next_image_path, terminal_reward + self.get_reward(next_state), True

	next_state = self.cur_state + input_action
	next_state = round(next_state, 2)
	next_image_path = self.dic[next_state]
	self.cur_state = next_state # state transfer
	return next_state, next_image_path, self.get_reward(next_state), False

    def import_angle(self):
	with open(self.angle_path, "r") as txtData:
	    lineData = txtData.readline()
	    Data = lineData.split() # first row
	# These are unchanged standard used in the test
	self.terminal_angle_low = float(Data[0])
	self.terminal_angle_high = float(Data[1])
	# print(Data[0], Data[1])
	return

    '''
    get_reward - reward determination
    '''
    def get_reward(self, next_angle):
	reward = 0 # suppose that it reaches the terminal range
	if next_angle < self.terminal_angle_low:
		reward = next_angle - self.terminal_angle_low
	elif next_angle > self.terminal_angle_high:
		reward = self.terminal_angle_high - next_angle
	return reward

    '''
    step - regulations of transfering between states in testing

    Input: input_action
    Return: next_state - the total angle of next state
    	    next_image_path - the image path of next state(the dictionary is not accessible from outside)
    	    terminal - True or False
    	    success - Success or not
    '''
    def test_step(self, input_action): # action is the angle to move
    	self.cur_step = self.cur_step + 1
    	# special case #1
    	if self.cur_state + input_action > MAX_ANGLE_LIMIT:
    	    return MAX_ANGLE, self.dic[MAX_ANGLE], True, False
    	# special case #2
    	if self.cur_state + input_action < MIN_ANGLE_LIMIT:
    	    return MIN_ANGLE, self.dic[MIN_ANGLE], True, False
	# special case #3
	if self.cur_step >= MAX_STEPS:
	    next_state = self.cur_state + input_action
	    next_state = round(next_state, 2)
	    next_image_path = self.dic[next_state]
	    # check for the final result: success or not
	    if next_state >= self.terminal_angle_low and next_state <= self.terminal_angle_high:
	    	return next_state, next_image_path, True, True
	    return next_state, next_image_path, True, False

	# get the next state, no need to calculate reward
	next_state = self.cur_state + input_action
	next_state = round(next_state, 2)
	next_image_path = self.dic[next_state]
	self.cur_state = next_state # state transfer
	return next_state, next_image_path, False, False
    
    '''
    import_dic - import the dictionary and return it

    Input: None
    Return: the imported dictionary
    '''
    def import_dic(self):
        with open(self.dict_path, "r") as txtData:
	    for lines in txtData:
	        lineData = lines.split()
	        self.dic[float(lineData[0])] = lineData[1]
        # print(self.dic)
        return self.dic
    
########################################################################
# Main - test the environment
########################################################################
if __name__ == '__main__':
	DICT_PATH = '/home/robot/RL/grp1/dict.txt'
	ANGLE_LIMIT_PATH = '/home/robot/RL/grp1/angle.txt'
	myenv = FocusEnv()
	cur_state, cur_image = myenv.reset(DICT_PATH, ANGLE_LIMIT_PATH)
	rList = []
	while True:
		ac = COARSE_POS
		next_state, next_image, r, t = myenv.step(ac)

		if t: break
		print "%f->%f, a:%f, r:%f, t:%d, %s" %(cur_state, next_state, ac, r, t, next_image)
		rList.append(r)
		cur_state = next_state
		time.sleep(0.2)

	plt.plot(rList, linewidth='2')
	plt.show()
