#######################################################################
# This file is for collecting virtual training environment, new version
# No dictionary and no terminal angle. But has angleList.
# Modified by xfyu on Apr 6
#######################################################################
# -*- coding: utf-8 -*-
# !/usr/bin/python
import pycontrol as ur
import numpy as np
import time
from ctypes import *
import cv2
from collections import deque
import os
import matplotlib.pyplot as plt

#######################################################################
# Global Variables
#######################################################################
# TIMES = [8.78, 1] # number of multiples
# STEP = [0.3, 0.1] # unit is radian, two levels
# ACTIONS = [-0.3*8.78, -0.3, 0.3, 0.3*8.78]
# maximum and minimum limitations
MIN_ANGLE_LIMIT = 0
MAX_ANGLE_LIMIT = 100
# CHANGE_POINT_LOW = 40.0
# CHANGE_POINT_HIGH = 60.0
ROOT_DIR = '/home/robot/RL/data'
GRP_NAME = 'new_grp3'
# path to store information
IMAGE_PATH = os.path.join(ROOT_DIR, GRP_NAME) # place to store the images
if not os.path.isdir(IMAGE_PATH):
	os.makedirs(IMAGE_PATH)

ANGLE_PATH = os.path.join(IMAGE_PATH, 'angle.txt')
angleList = [] # angle list
step = 0.3
'''
collect_env - collect images at all the possible states
'''
def collect_env():
	pic_num = 0
	cur_angle = 0.0
	while cur_angle >= MIN_ANGLE_LIMIT and cur_angle <= MAX_ANGLE_LIMIT:
		# take a pic
		pic_name = os.path.join(IMAGE_PATH, str(cur_angle) + '.jpg')
		angleList.append(cur_angle) # update angleList
		ur.camera_take_pic(pic_name)
		print("PIC NUM", pic_num, "/ CUR ANGLE", cur_angle, \
			"/ PIC NAME", pic_name)
		# update
		cur_angle = cur_angle + step
		cur_angle = round(cur_angle, 2)
		ur.change_focus_mode(ur.FINE)
		ur.move_from_to(step)

		# updata pic_num
		pic_num += 1
	
'''
showfocus - using TENEGRAD on every picture and plot a curve
'''
def showfocus():
	global angleList
	fList = [] # to store the curve
	for i in range(len(angleList)):
		pic_name = os.path.join(IMAGE_PATH, str(angleList[i])+ '.jpg')
		img = cv2.imread(pic_name)
		img = cv2.cvtColor(cv2.resize(img, (128,128)), cv2.COLOR_BGR2GRAY)
		fList.append(TENG(img))
		print("process", pic_name)
	plt.plot(angleList, fList, 'bx--')
	plot_name = GRP_NAME + '_focus'
	plt.savefig(os.path.join(IMAGE_PATH, plot_name), dpi=1200)
	plt.show()

	# find terminal angles
	max_focus = max(fList)
	print("max focus is %f" %(max_focus))
	flag_low = False
	for i in range(1, len(fList)):
		if flag_low == False and fList[i] >= 0.9 * max_focus:
			terminal_angle_low = angleList[i]
			flag_low = True
		if flag_low == True and fList[i] < 0.9 * max_focus:
			terminal_angle_high = angleList[i-1]
			break
    	print("terminal low is %f, terminal high is %f" %(terminal_angle_low, terminal_angle_high))
    	with open(ANGLE_PATH, 'w') as f:
    		Data = str(terminal_angle_low) + ' ' + str(terminal_angle_high)
    		f.write(Data)
    	return

'''
TENG - TENENGRAD
'''
def TENG(img):
    guassianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    guassianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return np.mean(guassianX * guassianX + guassianY * guassianY)

#######################################################################
# Main Process
#######################################################################
if __name__ == '__main__':
	# environment initialization
	ur.system_init()
	print("System Ready!")
	# collect env
	collect_env()
	
	# global angleList
	# angleList = [round(x*0.3, 2) for x in range(334)]
	# show focus curve by tenengrad
	showfocus()
	# move back to the init position
	ur.change_focus_mode(ur.COARSE)
	ur.move_from_to(-11.1)
	ur.gripper_open()
	ur.camera_close()
