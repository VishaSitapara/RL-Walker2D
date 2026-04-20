# Learning Bipedal Walking with Reinforcement Learning (Walker2d)

## Introduction: 
This project focuses on applying reinforcement learning (RL) to a continuous control problem using the Walker2d environment from Gymnasium, which is powered by the **MuJoCo**. The objective is to train an agent to learn stable and efficient walking behavior by controlling joint torques.

## Methodology:
The Walker2d environment models a two-dimensional bipedal robot to walk forward by controlling joint torques. The agent will observe states such as joint positions and velocities and take continuous actions to maximize cumulative reward. Experiments will include comparing algorithms (such as **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)**), analyzing reward trends, and studying the effects of hyperparameter tuning and reward design.

## Possible Results:
The RL agent is expected to learn stable walking patterns over time. The project will demonstrate differences in learning efficiency, convergence, and stability between algorithms.
