##################################################################
# This file is the real training environment.
# Include the state transfer. All the image processing are not here.
# Modified by xfyu on May 19, 2018.
##################################################################
#!/usr/bin/env python
import os
import random
import time
import cv2
import numpy
import collect_code.pycontrol as ur
#################################################################
# Important global parameters
#################################################################
# PATH = "/home/robot/RL" # current working path
PATH = os.path.split(os.path.realpath(__file__))[0]
REFERENCE_PATH = "/home/robot/RL/data/new_grp1" # the virtual env for max focus
# IMAGE_PATH = '/home/robot/RL/grp1'
SUCCESS_REWARD = 100
FAILURE_REWARD = -100
FOCUS_REWARD_NORM = 5.0
# ACTION_REWARD = 1
MAX_STEPS = 20
# maximum and minimum limitations, a little different from collectenv.py
# only part of the data is used: from 150.jpg to 180.jpg
MAX_ANGLE = 69.0
MIN_ANGLE = 30.0
CHANGE_POINT_RANGE = 1.5
# VERY IMPORTANT!!!
TIMES = 9
# actions
COARSE_POS = 0.3*TIMES
FINE_POS = 0.3
TERMINAL = 0
FINE_NEG = -0.3
COARSE_NEG = -0.3*TIMES


class FocusEnv():
    def __init__(self, SAVE_PIC_PATH):
	# COARSE NEG, FINE NEG, TERMINAL, FINE POS, COARSE POS
	self.actions = [COARSE_NEG, FINE_NEG, TERMINAL, FINE_POS, COARSE_POS]
	self.cur_state = 0.0 # initial with 0
	self.episode = 0
	self.save_pic_path = SAVE_PIC_PATH
	self.max_focus = self.get_max_focus()
	# the terminal angle should be acknouwledged during the training process
	ur.system_init()

    def __del__(self):
	ur.system_close()

    def reset(self):
    	# record the final state of last episode
    	last_state = self.cur_state
    	last_state = round(last_state, 2) # just in case
	# randomly decide the new initial state, the angle here is not accurate, just for random actions
	state = random.random() * (MAX_ANGLE - MIN_ANGLE)
	self.cur_state = MIN_ANGLE + state
	self.cur_state = round(self.cur_state, 2)
	self.cur_step = 0
	self.episode = self.episode + 1
	# fugure out the place to save pic
	self.save_pic_dir = os.path.join(self.save_pic_path, str(self.episode))
	if not os.path.isdir(self.save_pic_dir):
		os.makedirs(self.save_pic_dir)
	pic_name = str(self.cur_step) + '_' + str(self.cur_state) + '.jpg'
	pic_name = os.path.join(self.save_pic_dir, pic_name)
	init_path = pic_name
	# move from 0 to the initial state and take a pic
	# return the angle of first state and the name of the pic
	print("init angle:", self.cur_state)
	self.move(last_state, self.cur_state)
	ur.camera_take_pic(pic_name)

	# define the last focus
	img = cv2.imread(pic_name)
	self.last_focus = TENG(img)
	return self.cur_state, init_path

    '''
    step - regulations of transfering between states

    Input: input_action
    Return: next_state - the total angle of next state
    	    next_image_path - the image path of next state(the dictionary is not accessible from outside)
    	    reward - reward of this single operation
    	    terminal - True or False
    '''
    def step(self, input_action): # action is the angle to move
    	self.cur_step = self.cur_step + 1
    	# fugure out the place to save pic
    	pic_name = str(self.cur_step) + '_' + str(self.cur_state) + '.jpg'
    	pic_name = os.path.join(self.save_pic_dir, pic_name)
    	last_state = self.cur_state
    	last_state = round(last_state, 2) # just in case
    	# update self.cur_state 
	self.cur_state = self.cur_state + input_action
        self.cur_state = round(self.cur_state, 2)
        # move to the next state and take a pic
        if abs(input_action) > 1: # COARSE
        	ur.change_focus_mode(ur.COARSE)
        else: # fine
        	ur.change_focus_mode(ur.FINE)
        if input_action > 0: # UP
        	ur.send_movej_screw(ur.UP)
        elif input_action < 0: # DOWN
        	ur.send_movej_screw(ur.DOWN)
        ur.camera_take_pic(pic_name)
        next_image_path = pic_name

    	# special termination
    	if self.cur_state > MAX_ANGLE or self.cur_state < MIN_ANGLE:
    	    	return self.cur_state, next_image_path, FAILURE_REWARD + self.get_reward(pic_name), True

	# choose to terminate
	if input_action == TERMINAL:
	    	if self.cur_state >= self.terminal_angle_low and self.cur_state <= self.terminal_angle_high:
			return self.cur_state, next_image_path, SUCCESS_REWARD, True
	    	return self.cur_state, next_image_path, FAILURE_REWARD + self.get_reward(pic_name), True

	# special case - failure
	if self.cur_step >= MAX_STEPS:
	    	return self.cur_state, next_image_path, FAILURE_REWARD + self.get_reward(pic_name), True

	return self.cur_state, next_image_path, self.get_reward(pic_name), False

    '''
    get_reward - reward determination
    generate reward from the current taking pictures, should be less than 0
    '''
    def get_reward(self, pic_name):
	pic = cv2.imread(pic_name)
	cur_focus = TENG(pic)
	reward = (self.last_focus - cur_focus) * FOCUS_REWARD_NORM / self.max_focus
	print("last", self.last_focus, "cur", cur_focus, "reward is", reward)
	self.last_focus = cur_focus # update
	return reward # return

    '''
    move - move the ur3 from start_angle to end_angle
    '''
    def move(self, start_angle, end_angle):
	print("move from", start_angle, "to", end_angle)
    	if abs(end_angle - start_angle) > 10:
		delta_angle = (end_angle - start_angle) / TIMES
                delta_angle = round(delta_angle, 3)
    		print("COARSE move", delta_angle)
    		ur.change_focus_mode(ur.COARSE)
    		ur.move_from_to(delta_angle)
    	else:
		delta_angle = round(end_angle - start_angle, 3)
		print("FINE move", delta_angle)
		ur.change_focus_mode(ur.FINE)
		ur.move_from_to(delta_angle)

    '''
    coarse_move - move in coarse, from start_angle to end_angle
    '''
    def coarse_move(self, start_angle, end_angle):
        print("coarse move from", start_angle, "to", end_angle)
        delta_angle = end_angle - start_angle
        delta_angle = round(delta_angle, 3)
        print("COARSE move", delta_angle)
        ur.change_focus_mode(ur.COARSE)
        ur.move_from_to(delta_angle)

    '''
    get_max_focus - get the max focus for reference
    '''
    def get_max_focus(self):
    	cur_angle = MIN_ANGLE
    	max_focus = 0.0
    	while cur_angle <= MAX_ANGLE:
    		pic_name = os.path.join(REFERENCE_PATH, str(cur_angle)+'.jpg')
    		img = cv2.imread(pic_name)
    		cur_focus = TENG(img)
    		if cur_focus > max_focus:
    			max_focus = cur_focus
    		cur_angle += 0.3
    	return max_focus


def TENG(img):
    guassianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    guassianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return numpy.mean(guassianX * guassianX + 
    				  guassianY * guassianY)

if __name__ == '__main__':
	save_pic_dir = '/home/robot/RL/training/realenv_traintest'
	env = FocusEnv(save_pic_dir)
        env.move(0, 49.0)
        cur_angle = 49.0
        while (cur_angle < 54.0):
        	my_pic = os.path.join(save_pic_dir, str(cur_angle)+'.jpg')
        	ur.camera_take_pic(my_pic)
        	# refer_pic = os.path.join("/home/robot/RL/data/new_grp1", str(cur_angle)+'.jpg')
        
        	my_pic_img = cv2.imread(my_pic)
        	# refer_pic_img = cv2.imread(refer_pic)
        	# print("my pic:", TENG(my_pic_img), "refer pic:", TENG(refer_pic_img))
		print("my pic:", TENG(my_pic_img))
		env.move(cur_angle, cur_angle+0.3)
		cur_angle += 0.3
	env.move(cur_angle, 0)
