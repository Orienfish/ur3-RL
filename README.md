# Applying Deep Reinforcement Learning to Auto-Focus with UR3 Robotic Arm
Authors: Xiaofan Yu, Runze Yu, Jingsong Yang and Xiaohui Duan. <br>
This repo includes the key implementation of our paper "A Robotic Auto-Focus System based on Deep Reinforcement Learning" to be appeared at ICARCV 2018. The link to our paper will be added later. <br>

## Overview
Auto-focus has wide applications in diseases diagnoises. The contribution of our work can be summarized as follows:
1. We introduce an end-to-end method that directly processes vision input (microscopic view) and selects an optimal action. Previous solutions to auto-focus problems contains two distinct stages: the focus measure functions that map an image to a value for representing the degree of focus of the image, and the search algorithms that iteratively move the lens to find the highest or nearest peak of focus measure curves. Here in our paper, we combine those two stages into one, which enables the agent to complete "perception" and "control" as a whole. 
2. We formulate the auto-focus problem into a model-free decision-making task analogous to Atari games: we discretize state and action space to apply Deep Q Network (DQN). DQN directly processes high-dimensional input and learns policies through trials and errors. This could be a general approach in vision-based control problems.
3. We design the reward function according to focus measure functions and more importantly, we define an active termination condition which expects the agent to stop at a clear spot automaticly.

The model of our system and hardware implementations are shown in the following figures. <br>
<div align=center><img width="800" height="250" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/model%26imple.png"/></div>

As for experiments, the training progress are categorized into two parts: virtual training and real training. The motivation of this setting is to save time in real scenario as operating real robots takes unbearable time. Virtual experiments, which are carried out after the virtual training phase, indicates that our method could achieve 100% accuracy on a certain view with different focus range. Further training on real robots could eliminate the deviation between the simulator and real scenario, leading to reliable performances in real applications. <br>

A typical virtual learning curve after 100K episodes is shown as following. <br>
<div align=center><img width="450" height="150" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/vexp1_up.PNG"/></div>
Distribution of focus positions regarding both the virtually-trained model and real-trained model is depicted in the following figure. We can see that while the virtually-trained model still finishes focusing with failure from time to time, the focused positions of the practically-trained model move rightwards and all locate in a clear enough area for human vision, indicating a 100\% accuracy in real auto-focus. <br>
<div align=center><img width="450" height="450" src="https://github.com/Orienfish/ur3-RL/blob/master/pic/endf.png"/></div>