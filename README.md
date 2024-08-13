# 【Mathematical Foundation of Reinforcement Learning】 Coding Exercise and Notes

## Introduction

Thanks to the video courses *[Mathematical Foundation of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main)*, I have a better understanding of the mathematical foundation of reinforcement learning. This repository contains my coding exercises and notes which might be a proper coding tutorial for the beginners both for the principles and coding, because only the crucial parts in each algorithm are designed to be blank in order to help beginners better understand by realization while reducing the burden of programming.

Apart from the essential coding exercise, the notes contained in the folder _[Algorithms](https://github.com/SupermanCaozh/RL_Coding_Exercise/tree/master/Algorithms)_, which are not simply copied from the textbook, are attached to make some concepts as well as the easy-to-confuse parts more understandable. The notes are written in Chinese temporarily, and will be translated into English in the future.

Thanks to the existing codes of the '[Grid World](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main/Code%20for%20grid%20world/python_version/src)' environment from the teaching assistant, which saved a lot of time for me to focus on the core part of these classical algorithms. The necessity to create a proper and reasonable environment is as important as the algorithms themselves.

## What does this repo contain？

This repository contains the hero algorithms in each chapter in the book except Chapter 6 which gives the mathmatical foundation of temporal difference algorithms but is very important. There are two fundamental ways to category these algorithms. One cuts in from the perspective of the accessibility of the model, while the other origins from the view of the difference between the sampling policy, namely the behavior policy, and the target policy.

The *model-based* algorithms include *Value Iteration* and *Policy Iteration*, in which the model is known a prior to the user. In the field of *model-free* algorithms, there are *Monte-Carlo*, *Sarsa*, *Q-Learining(Deep Q-Learning)*, *Policy Gradient* and *A2C* with their variants. More advanced RL algorithms will be updated here in the future...
