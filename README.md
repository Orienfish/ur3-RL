# A Robotic Auto-Focus System based on Reinforcement Learning
Authors: Xiaofan Yu, Runze Yu, Jingsong Yang and Xiaohui Duan. <br>
This repo includes the key implementation of our paper "A Robotic Auto-Focus System based on Deep Reinforcement Learning" to appear at ICARCV 2018. You can find our contribution [here](https://arxiv.org/abs/1809.03314). <br>

## Overview
Auto-focus has wide applications in diseases diagnoises. The contribution of our work can be summarized as follows:
1. We introduce an end-to-end method that directly processes vision input (microscopic view) and selects an optimal action to screw the focusing knob. Previous solutions to auto-focus problems contains two distinct stages: the focus measure functions that map an image to a value for representing the degree of focus of the image, and the search algorithms that iteratively move the lens to find the highest or nearest peak of focus measure curves. Here in our paper, we combine those two stages into one, which enables the agent to complete "perception" and "control" as a whole. 
2. We formulate the auto-focus problem into a model-free decision-making task analogous to Atari games: we discretize state and action space to apply Deep Q Network (DQN). DQN directly processes high-dimensional input and learns policies through trials and errors. This could be a general approach in vision-based control problems.
3. We design the reward function according to focus measure functions and more importantly, we define an active termination condition which expects the agent to stop at a clear spot automaticly.

The model of our system and hardware implementations are shown in the following figures. <br>
<div align=center><img width="800" height="250" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/model%26imple.png"/></div>

As for experiments, the training progress are categorized into two parts: virtual training and real training. The motivation of this setting is to save time in real scenario as operating real robots takes unbearable time. Virtual experiments, which are carried out after the virtual training phase, indicates that our method could achieve 100% accuracy on a certain view with different focus range. Further training on real robots could eliminate the deviation between the simulator and real scenario, leading to reliable performances in real applications. <br>

A typical virtual learning curve after 100K episodes is shown as following. <br>
<div align=center><img width="450" height="150" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/vexp1_up.PNG"/></div>
Distribution of focus positions regarding both the virtually-trained model and real-trained model is depicted in the following figure. We can see that while the virtually-trained model still finishes focusing with failure from time to time, the focused positions of the practically-trained model move rightwards and all locate in a clear enough area for human vision, indicating a 100% accuracy in real auto-focus. <br>
<div align=center><img width="350" height="280" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/endf.png"/></div>

## File Structure
```
├── README.md                 // Help
├── collect              	  // Code used in collecting data to construct virtual environment.
│   ├── collectenv_new.py     // Python code for collecting data.
│   ├── pycontrol.py          // Python interface for controling microscope and camera.
│   ├── lib.so                // Generated dynamic linking library from C. To control UR3 and gripper.
│   ├── modbustcp.c/.h        // Low-level code written in C to control UR3.
│   ├── modbusrtu.c/.h        // Low-level code written in C to control gripper.
│   ├── init_pos.txt       	  // Record the settled initial position of UR3.
│   ├── camera_lib.so         // Generated dynamic linking library from C. To control camera.
│   ├── qhyccd*.h/camera.cpp  // Low-level code written in C++ to control camera.
│   ├── main.h                // Necessary macro definitions.
│   └── __init__.py           // Empty file. For wrapping the whole file as a python package.
├── deep_q_network_virfnew.py // DQN for virtual training and simutaneous testing.
├── trainenv_virf_v5.py       // Python class for virtual training environment. Interact with deep_q_network_virf_new.py.
├── test_model_v.py           // Extra virtual test after virtual training.
├── deep_q_network_real_train.py // DQN for practical training.
├── realenv_train.py          // Python class for real training environment. Interact with deep_q_network_real_train.py.
├── deep_q_network_real_test.py // DQN for practical testing.
└── realenv_test.py           // Python class for real testing environment. Interact with deep_q_network_real_test.py. 
```
As you can see, all the files can be divided into two groups: one is for **Reinforcement Learning (RL)** in both virtual and practical environment, the other is for **collecting data** to construct virtual environment. The RL group locates right under the root directory while the collecting code group is under "collect" directory. <br>

* What we mean by **"collecting data"** here is sampling discrete microscopic views with a fixed step and number them with the absolution angle of focusing knob. To help you get an intuitive idea of what we collect here, I attached a focus measure curve of all the views here: <br>
<div align=center><img width="350" height="280" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/new_grp1_focus.png"/></div>
It's important to make sure that we get the complete region from blur to clear and to blur again. Due to the large file size, unfortunately, I didn't upload the dateset we collected here on Github. Feel free to contact me if you are interested in those data. <br> <br>

* The **RL** code mainly constructs a *DQN* and triggers as well as monitors the learning process. The structure of our DQN is shown in the following figure. The whole network contains 381K parameters and requires 13.8M multiply-accumulate operations in each update (if I'm not making calculation errors lol).
<div align=center><img width="800" height="250" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/network.png"/></div>

## Environment
* PC with 8 Intel(R) Xeon(R) E5-1620 @3.50GHz CPUs, a 64GB RAM and 2 NVIDIA Quadro M4000 8GB GPUs <br>
Ubuntu 16.04 LTS <br>
Python 2.7.12 <br>
tensorflow 1.6.0 <br>
CUDA 9.0.176 <br>
CuDNN 7.0 <br>

* [UR3 Robotic Arm](https://www.universal-robots.com/products/ur3-robot/)
* [ROBOTIQ 2-Finger Adaptive Robot Gripper](https://www.universal-robots.com/plus/end-effectors/robotiq-2-finger-adaptive-robot-gripper/)
* [XSZ-4GA Optical Microscope](https://abbey.en.alibaba.com/product/60684855332-806650703/XSZ_4GA_1600X_on_surgical_a_high_quality_operating_Binocular_Biological_Microscope.html)

## Version Record
v1 @9/3/2018 <br>
Uploaded before submitting the camera-ready version of our paper.