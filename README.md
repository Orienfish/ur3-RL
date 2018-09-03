# Applying Deep Reinforcement Learning to Auto-Focus with UR3 Robotic Arm
Authors: Xiaofan Yu, Runze Yu, Jingsong Yang and Xiaohui Duan. <br>
This repo includes the key implementation of our paper "A Robotic Auto-Focus System based on Deep Reinforcement Learning" to be appeared at ICARCV 2018. The link to our paper will be added later. <br>

## Background
Auto-focus has wide applications in diseases diagnoises. The contribution of our work can be summarized as follows:
1. We introduce an end-to-end method that directly processes vision input (microscopic view) and selects an optimal action. Previous solutions to auto-focus problems contains two distinct stages: the focus measure functions that map an image to a value for representing the degree of focus of the image, and the search algorithms that iteratively move the lens to find the highest or nearest peak of focus measure curves. Here in our paper, we combine those two stages into one, which enables the agent to complete "perception" and "control" as a whole. 
2. We formulate the auto-focus problem into a model-free decision-making task analogous to Atari games: we discretize state and action space to apply Deep Q Network (DQN). DQN directly processes high-dimensional input and learns policies through trials and errors. This could be a general approach in vision-based control problems.
3. We design the reward function according to focus measure functions and more importantly, we define an active termination condition which expects the agent to stop at a clear spot automaticly.

The model of our system in real applications is shown in the following figure. <br>
<div align=center><img width="600" height="500" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/systemmodel.PNG"/></div>