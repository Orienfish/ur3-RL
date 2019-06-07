##################################################################
# This file is the real training environment.
# Include the state transfer. All the image processing are not here.
# Modified by xfyu on May 19, 2018.
##################################################################
# !/usr/bin/env python
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
# IMAGE_PATH = '/home/robot/RL/grp1'
# MAX_STEPS = 20
# maximum and minimum limitations, a little different from collectenv.py
# only part of the data is used: from 150.jpg to 180.jpg
# MAX_ANGLE = 69.0
# MIN_ANGLE = 30.0
SUCCESS_REWARD = 100
FAILURE_REWARD = -100
FOCUS_REWARD_NORM = 10.0
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
# pic resize
RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128


class FocusEnvTest():  # one class for one folder
    def __init__(self, info, pathinfo):  # info:[MAX_STEPS, MIN_ANGLE, MAX_ANGLE]
        # pathinfo:[MAIN_DIR, REFERENCE_DIR]
        # COARSE NEG, FINE NEG, TERMINAL, FINE POS, COARSE POS
        self.actions = [COARSE_NEG, FINE_NEG, TERMINAL, FINE_POS, COARSE_POS]
        self.max_steps, self.min_angle, self.max_angle = info[0], info[1], info[2]
        self.main_dir, self.reference_path = pathinfo[0], pathinfo[1]

        self.cur_state = 0.0  # initial with 0.0 
        self.episode = 0  # init episode num
        self.get_max_focus()  # set focus points

        # ur.system_init() # If call test with train at the same time, annotate this sentence out

    def __del__(self):
        # ur.system_close()  # If call test with train at the same time, annotate this sentence out
        pass

    def reset(self, other_env_state=-1.0,
              test_cnt=0):  # -1.0 an impossible value, test_cnt represents the index of current test
        # record the final state of last episode or last state of the other env
        if other_env_state == -1.0:  # reload last state from the same env
            last_state = self.cur_state
            last_state = round(last_state, 2)  # just in case
        else:
            last_state = round(other_env_state, 2)  # init a new test, reload last state from the other env
            self.save_pic_path = os.path.join(self.main_dir, str(test_cnt))
            if os.path.isdir(self.save_pic_path):  # if already exists, delete
                shutil.rmtree(self.save_pic_path)
            os.makedirs(self.save_pic_path)  # new it
            self.episode = 0  # reset to 0
        # randomly decide the new initial state, the angle here is not accurate, just for random actions
        state = random.random() * (self.max_angle - self.min_angle)
        self.cur_state = self.min_angle + state
        self.cur_state = round(self.cur_state, 2)
        self.cur_step = 0
        self.episode = self.episode + 1

        # fugure out the place to save pic
        # use separate dirs under self.save_pic_path
        self.episode_dir = os.path.join(self.save_pic_path, str(self.episode))
        if not os.path.isdir(self.episode_dir):
            os.makedirs(self.episode_dir)
        # generate first image path
        pic_name = str(self.cur_step) + '_' + str(self.cur_state) + '.jpg'
        pic_name = os.path.join(self.episode_dir, pic_name)
        init_path = pic_name
        # move from 0 to the initial state and take a pic
        # return the angle of first state and the name of the pic
        print("init angle:", self.cur_state)
        self.move(last_state, self.cur_state)
        ur.camera_take_pic(pic_name)

        return self.cur_state, init_path

    '''
    step - regulations of transfering between states
    Input: input_action
    Return: next_state - the total angle of next state
            next_image_path - the image path of next state(the dictionary is not accessible from outside)
            reward - reward of this single operation
            terminal - True or False
    '''

    def test_step(self, input_action):  # action is the angle to move
        self.cur_step = self.cur_step + 1
        # fugure out the place to save pic
        pic_name = str(self.cur_step) + '_' + str(self.cur_state) + '.jpg'
        pic_name = os.path.join(self.episode_dir, pic_name)
        # update self.cur_state 
        self.cur_state = self.cur_state + input_action
        self.cur_state = round(self.cur_state, 2)

        # real movement
        # move to the next state and take a pic
        if abs(input_action) > 1:  # COARSE
            ur.change_focus_mode(ur.COARSE)
        elif abs(input_action) > 0.0:  # fine
            ur.change_focus_mode(ur.FINE)
        if input_action > 0:  # UP
            ur.send_movej_screw(ur.UP)
        elif input_action < 0:  # DOWN
            ur.send_movej_screw(ur.DOWN)
        ur.camera_take_pic(pic_name)
        next_image_path = pic_name

        # special termination
        if self.cur_state > self.max_angle or self.cur_state < self.min_angle:
            return self.cur_state, next_image_path, True, False

        # choose to terminate
        if input_action == TERMINAL:
            focus = TENG(pic_name)
            if focus > self.success_focus:
                return self.cur_state, next_image_path, True, True  # success!
            return self.cur_state, next_image_path, True, False

        # special case - failure
        if self.cur_step >= self.max_steps:
            return self.cur_state, next_image_path, True, False

        return self.cur_state, next_image_path, False, False

    '''
    get_max_focus - get the max focus for reference
    '''

    def get_max_focus(self):
        cur_angle = self.min_angle
        max_focus = 0.0
        # walk through every images in the directory
        while cur_angle <= self.max_angle:
            pic_name = os.path.join(self.reference_path, str(cur_angle) + '.jpg')
            print("calculating %s" % pic_name)
            # read and calculate focus
            cur_focus = TENG(pic_name)
            if cur_focus > max_focus:
                max_focus = cur_focus
            cur_angle += 0.3  # move to the next picture
        print("max focus is", max_focus)
        self.change_point_focus = max_focus * CHANGE_POINT_THRES
        self.success_focus = max_focus * SUCCESS_THRES
        self.max_focus = max_focus
        return max_focus

    '''
    move - move the ur3 from start_angle to end_angle
    '''

    def move(self, start_angle, end_angle):
        print("move from", start_angle, "to", end_angle)
        if abs(end_angle - start_angle) > 10:
            delta_angle = (end_angle - start_angle) / TIMES
            delta_angle = round(delta_angle, 2)
            print("COARSE move", delta_angle)
            ur.change_focus_mode(ur.COARSE)
            ur.move_from_to(delta_angle)
        else:
            delta_angle = round(end_angle - start_angle, 2)
            print("FINE move", delta_angle)
            ur.change_focus_mode(ur.FINE)
            ur.move_from_to(delta_angle)


'''
Tenengrad
'''


def TENG(img_path):
    img = cv2.imread(img_path)  # read pic
    img = cv2.cvtColor(cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT)), cv2.COLOR_BGR2GRAY)  # resize
    guassianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    guassianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return numpy.mean(guassianX * guassianX + guassianY * guassianY)
