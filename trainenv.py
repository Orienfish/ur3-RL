##################################################################
# This file is the virtual training environment.
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
CHANGE_POINT_FINAL = 1.1
CHANGE_POINT_INITIAL = 5
EXTRA_RANGE_INITIAL = 20
SUCCESS_REWARD = 1000
FAILURE_REWARD = -100
POSITIVE_REWARD_STANDARD = 1
NEGATIVE_REWARD_STANDARD = -0.1
MAX_STEPS = 100
# maximum and minimum limitations, a little different from collectenv.py
MAX_ANGLE = 99.9
MIN_ANGLE = 0.0
MIN_ANGLE_LIMIT = 0.3
MAX_ANGLE_LIMIT = 99.6
# actions
COARSE_POS = 0.3*9
FINE_POS = 0.3
TER = 0
FINE_NEG = -0.3
COARSE_NEG = -0.3*9

class FocusEnv(): # one class for one folder
    def __init__(self, DICT_PATH, ANGLE_LIMIT_PATH):
	# define the action space
	self.actions = [COARSE_NEG, FINE_NEG, TER, FINE_POS, COARSE_POS]
	self.dic = dict()
	self.dict_path = DICT_PATH
    	self.angle_path = ANGLE_LIMIT_PATH
	self.import_dic()
	self.import_angle()
	# init the success and fail cnt to let the user know the result, init only once
	self.success_cnt = 0
	self.fail_cnt = 0
	# set the parameters, init only once
	self.change_point_t = CHANGE_POINT_INITIAL
	self.extra_range = EXTRA_RANGE_INITIAL

    def reset(self): # reset starts a new episode, never change DICT_PATH and ANGLE_LIMIT_PATH
    # reset the starting state
	# index = int(random.random()*len(self.dic.keys())) # randomly initialized
	# start from 0
	self.cur_state = 0.0 # self.dic.keys()[index] 
	self.cur_step = 0
	# return the angle of first state and the image name
	return self.cur_state, self.dic[self.cur_state]

    '''
    step - regulations of transfering between states

    Input: input_action
    Return: next_state - the total angle of next state
    		next_image_path - the image path of next state(the dictionary is not accessible from outside)
    		reward - reward of this single operation
    		terminal - True or False
    '''
    def step(self, input_action, time_step, EXPLORE): # action is the angle to move
    	self.cur_step = self.cur_step + 1
    	# special case #1
    	if self.cur_state + input_action > MAX_ANGLE_LIMIT:
    	    return MAX_ANGLE, self.dic[MAX_ANGLE], FAILURE_REWARD, True, self.cur_step
    	# special case #2
    	if self.cur_state + input_action < MIN_ANGLE_LIMIT:
    	    return MIN_ANGLE, self.dic[MIN_ANGLE], FAILURE_REWARD, True, self.cur_step
	# special case #3
	if self.cur_step >= MAX_STEPS:
	    # self.reset(self.dict_path, self.angle_path)
	    next_state = round(self.cur_state + input_action, 2)
	    next_image_path = self.dic[next_state]
	    return next_state, next_image_path, FAILURE_REWARD, True, self.cur_step

	# determine the terminal angle
	self.terminal_angle_determine(time_step, EXPLORE)

	# input action is terminal
	if input_action == TER:
		# reach the terminal zone, success
	    if self.cur_state >= self.terminal_angle_t_low and \
	    	self.cur_state <= self.terminal_angle_t_high:
	    	self.success_cnt += 1 # update
	    	return self.cur_state, self.dic[self.cur_state], SUCCESS_REWARD, True, self.cur_step
	    # outside the terminal zone, fail
	    else:
	    	self.fail_cnt += 1 # update
	    	return self.cur_state, self.dic[self.cur_state], FAILURE_REWARD, True, self.cur_step

	# normal cases: one right answer
	reward = NEGATIVE_REWARD_STANDARD
	# state 1
	if self.cur_state < self.change_point_t_low:
	    if input_action == COARSE_POS: # COARSE POS
	    	# print("COARSE POS")
	    	reward = POSITIVE_REWARD_STANDARD
	# state 2
	elif self.cur_state >= self.change_point_t_low and \
		self.cur_state < self.terminal_angle_t_low:
	    if input_action == FINE_POS: # FINE POS
	    	# print("FINE POS")
	    	reward = POSITIVE_REWARD_STANDARD
	# state 3 - already process in the above
	# elif self.cur_state >= self.terminal_angle_t_low and \
	#    self.cur_state <= self.terminal_angle_t_high:
	#    if input_action == TER: # TERMINAL
	#    	return self.cur_state, self.dic[self.cur_state], SUCCESS_REWARD, True
	# state 4
	elif self.cur_state > self.terminal_angle_t_high and \
	    self.cur_state < self.change_point_t_high:
	    if input_action == FINE_NEG: # FINE NEG
	    	# print("FINE NEG")
	    	reward = POSITIVE_REWARD_STANDARD
	elif self.cur_state >= self.change_point_t_high:
	    if input_action == COARSE_NEG: # COARSE NEG
	    	# print("COARSE NEG")
	    	reward = POSITIVE_REWARD_STANDARD
	# calculate the reward
	# reward = self.reward_multiply(reward) # calculate the multiply
	next_state = self.cur_state + input_action
	next_state = round(next_state, 2)
	next_image_path = self.dic[next_state]
	self.cur_state = next_state # state transfer
	return next_state, next_image_path, reward, False, self.cur_step

    def import_angle(self):
	with open(self.angle_path, "r") as txtData:
	    lineData = txtData.readline()
	    Data = lineData.split() # first row
	self.terminal_angle_low = float(Data[0])
	self.terminal_angle_high = float(Data[1])
	# print(Data[0], Data[1])
	return

    '''
    terminal_angle_determine - determine terminal_angle_t and change_angle_t
    '''
    def terminal_angle_determine(self, time_step, EXPLORE):
	# after the EXPLORE times should reach the final value
	if self.extra_range > 0:
		self.extra_range -= EXTRA_RANGE_INITIAL / EXPLORE
		self.terminal_angle_t_low = round(self.terminal_angle_low - self.extra_range, 2)
		self.terminal_angle_t_high = round(self.terminal_angle_high + self.extra_range, 2)
		print("terminal:", self.extra_range, self.terminal_angle_t_low, self.terminal_angle_t_high)
	if self.change_point_t > CHANGE_POINT_FINAL:
		self.change_point_t -= (CHANGE_POINT_INITIAL - CHANGE_POINT_FINAL) / EXPLORE
		self.change_point_t_low = round(self.terminal_angle_t_low - self.change_point_t, 2)
		self.change_point_t_high = round(self.terminal_angle_t_high + self.change_point_t, 2)
		print("change:", self.change_point_t, self.change_point_t_low, self.change_point_t_high)
	return


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
    
    '''
    reward_multiply - decrease the reward as the step increase
    '''
    '''
    def reward_multiply(self, reward):
    	if reward < 0:
    		return reward
    	else:
    		return reward * (1-0.0001*self.cur_step*self.cur_step)
	'''
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
