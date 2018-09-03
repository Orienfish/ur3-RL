#######################################################################
# This file is the python API to control UR3 robotic arm.
# Modified by xfyu on Apr 6
#######################################################################
# -*- coding: utf-8 -*-
# !/usr/bin/python
from ctypes import * 
import time
import os

REG_NUM = 6
FORCE = 20
SPEED = 40

# FOCUS MODE
FOCUS_STATUS = 0
COARSE = 0
FINE = 1
COARSE_POS = (c_float*REG_NUM)(0.0,0.0,0.0,0.0,0.0,0.0)
FINE_POS = (c_float*REG_NUM)(0.0,0.0,0.0,0.0,0.0,0.0)

# screw direction
UP = 1
DOWN = -1

# init pos
INIT_POS_FILE ="/home/robot/RL/collect_code/init_pos.txt"
# import library
# gcc -o lib.so -fPIC -shared modbustcp.c modbusrtu.c
# g++ camera.cpp -o camera_lib.so -fPIC -shared `pkg-config --cflags opencv``pkg-config --libs opencv` -I /usr/local/include -L /usr/local/lib -lqhy
PATH = os.path.split(os.path.realpath(__file__))[0]
lib = cdll.LoadLibrary(os.path.join(PATH, "lib.so"))
camera_lib = cdll.LoadLibrary(os.path.join(PATH, "camera_lib.so"))

#
# read_pos - Read 6 pos value
# p[x, y, z, rx, ry, rz]. The x, y, z are in 0.1mm base.
# The rx, ry, rz are in 0.001rad base. 
#
# Parameter: None
# Return value: success - 6 values in m base and rad base
#               fail - None
def read_pos():
	# set the type of parameters
    lib.read_pos.argtype = [POINTER(c_float),] 
    lib.read_pos.restype = c_int
    ret_value = (c_float * REG_NUM)()

  	# read pos position
    res = lib.read_pos(ret_value)
    if not res: # read successfully
    	return ret_value
    return None # fail

#
# read_wrist - Read 6 joint value
# [base, shoulder, elbow, wrist1, wrist2, wrist3].
# All values are in 0.001rad base. 
#
# Parameter: None
# Return value: success - 6 values in rad base
#               fail - None
#
def read_wrist():
	# set the type of parameters
    lib.read_wrist.argtype = [POINTER(c_float),] 
    lib.read_wrist.restype = c_int
    ret_value = (c_float * REG_NUM)()

  	# read pos position
    res = lib.read_wrist(ret_value)
    if not res: # read successfully
    	return ret_value
    return None # fail

#
# send_movel_instruct - send move intruction to robot
#						Using position.
#
# Input: desired_pose - the target position
# Return: None.
#         Error message will display in the Msg box.
#
def send_movel_instruct(desired_pose_pointer):
	# set the type of parameters
    lib.send_movel_instruct.argtype = [POINTER(c_float),]
    lib.send_movel_instruct.restype = c_int
    res = lib.send_movel_instruct(desired_pose_pointer)
    return res  

#
# send_movej_instruct - send move intruction to robot
#						Using joint positions.
# move to the position decribed by angles' values
#
# Input: float * desired_joint - target position
# Return: none.
#         Error message will display in the Msg box.
#
def send_movej_instruct(desired_joint_pointer):
	# set the type of parameters
    lib.send_movej_instruct.argtype = [POINTER(c_float),]
    lib.send_movej_instruct.restype = c_int
    res = lib.send_movej_instruct(desired_joint_pointer)
    return res

#
# gripper_activate - activate the gripper
#
# Speed and force are set in the above
# Return Value: 0 - success
#               -1 - fail
#
def gripper_activate():
	lib.gripper_activate.restype = c_int
	res = lib.gripper_activate()
	return res

#
# gripper_close - close the gripper
#
# Speed and force are set in the above
# Return Value: 0 - success
#               -1 - fail
#
def gripper_close():
	lib.gripper_close.argtype = [c_ubyte, c_ubyte]
	lib.gripper_close.restype = c_int
	res = lib.gripper_close(SPEED, FORCE)
	return res

#
# gripper_open - open the gripper
#
# Speed and force are set in the above
# Return Value: 0 - success
#               -1 - fail
#
def gripper_open():
	lib.gripper_close.argtype = [c_ubyte, c_ubyte]
	lib.gripper_close.restype = c_int
	res = lib.gripper_open(SPEED, FORCE)
	return res

#
# camera_init
#
def camera_init():
	camera_lib.camera_init()

#
# camera_close
#
def camera_close():
	camera_lib.camera_close()

#
# camera_take_pic
#
def camera_take_pic(pic_name):
	camera_lib.camera_take_pic.argtype = [c_char_p,]
	camera_lib.camera_take_pic(c_char_p(pic_name))

#
# read_init_pose - read initial position from txt file
#
def read_init_pose():
	with open(INIT_POS_FILE, "r") as txtData:
	    # read coarse pose
	    lineData = txtData.readline()
	    Data = lineData.split() # first row
	    for i in range(REG_NUM):
	    	COARSE_POS[i] = float(Data[i])

	    # read fine pose
	    lineData = txtData.readline()
	    Data = lineData.split() # second row
	    for i in range(REG_NUM):
	    	FINE_POS[i] = float(Data[i])


#
# change_focus_mode - change the focus mode to what you want
#
def change_focus_mode(TARGET_MODE):
	global FOCUS_STATUS
	# no need to change
	if FOCUS_STATUS == TARGET_MODE:
		return
	# change to COARSE
	if TARGET_MODE == COARSE:
		gripper_open()
		send_movej_instruct(byref(COARSE_POS))
		gripper_close()
		FOCUS_STATUS = COARSE
	# change to FINE
	else:
		gripper_open()
		send_movej_instruct(byref(FINE_POS))
		gripper_close()
		FOCUS_STATUS = FINE

#
# send_movej_screw
#
def send_movej_screw(direction):
	lib.send_movej_screw.argtype = c_int
	lib.send_movej_screw(direction)

#
# move - move from the start angle to the end angle
# Block the user away from the lower details
#
def move_from_to(next_move):
	direction = UP
	if next_move < 0:
		direction = DOWN
	next_move = abs(next_move)
	next_move = round(next_move, 3)
	lib.move_from_to.argtype = [c_int, c_float]
	lib.move_from_to(direction, c_float(next_move))
#
# system_init - initialization for collection
#
def system_init():
	gripper_activate()
	gripper_open()
	read_init_pose()
	send_movej_instruct(byref(COARSE_POS))
	STATUS = COARSE # start from COARSE
	gripper_close()
	camera_init()

#
# system_close - called after everything completes
#
def system_close():
	gripper_open()
	camera_close()


# main test
if __name__ == '__main__':
	system_init()
	move_from_to(0.0, 12.0)
	# gripper_activate()
	
	'''
	Test 1: read current position and send movl instruction
	'''
	# ret = read_pos()
	# print "%f %f %f %f %f %f" %(ret[0], ret[1], ret[2], ret[3], ret[4], ret[5])
	# ret[2] = ret[2] + 0.05
	# send_movel_instruct(byref(ret))
	# print "finish test 1"
	# time.sleep(3)

	'''
	Test 2: read current wrist angle and send movej instruction
	'''
	# ret = read_wrist()
	# print "%f %f %f %f %f %f" %(ret[0], ret[1], ret[2], ret[3], ret[4], ret[5])
	# ret[2] = ret[2] + 0.3
	# send_movej_instruct(byref(ret))
	# print "finish test 2"
	# time.sleep(3)

	'''
	Test 3: when reach minimum and maximum bound
	'''
	# target angle value
	# goal = (c_float*REG_NUM)(5.523,-1.145,0.812,0.815,0.455,0.054)
	# gripper_close()
	# send_movej_instruct(byref(goal))
	# send_movej_screw(DOWN)
	# send_movej_screw(UP)
	# send_movej_screw(UP)
