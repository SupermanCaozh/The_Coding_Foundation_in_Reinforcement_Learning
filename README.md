# 【Mathematical Foundation of Reinforcement Learning】 Coding Exercise and Notes

## Introduction

Thanks to the video courses *[Mathematical Foundation of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main)*, I have a better understanding of the mathematical foundation of reinforcement learning. This repository contains my coding exercises and notes which might be a proper coding tutorial for the beginners both for the principles and coding, because only the crucial parts in each algorithm are designed to be blank in order to help beginners better understand by realization while reducing the burden of programming.

Apart from the essential coding exercise, the notes contained in the folder _[Algorithms](https://github.com/SupermanCaozh/RL_Coding_Exercise/tree/master/Algorithms)_, which are not simply copied from the textbook, are attached to make some concepts as well as the easy-to-confuse parts more understandable. The notes are written in Chinese temporarily, and will be translated into English in the future.

Thanks to the existing codes of the '[Grid World](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main/Code%20for%20grid%20world/python_version/src)' environment from the teaching assistant, which saved a lot of time for me to focus on the core part of these classical algorithms. The necessity to create a proper and reasonable environment is as important as the algorithms themselves.

Apart from the envorinment above, some other classic environment in the filed of optimal control that are provided in the package _gym_ released by OpenAI are tested in these codes, especially in the scripts of _Policy Gradient_. There are some reasons for this 'rebellious behavior' here and the most important one is that not every algorithm outperform others in an environment with discrete action space or state space.

## What does this repo contain？

This repository contains the hero algorithms in each chapter in the book except Chapter 6 which gives the mathmatical foundation of temporal difference algorithms but is very important. There are two fundamental ways to category these algorithms. One cuts in from the perspective of the accessibility of the model, while the other origins from the view of the difference between the sampling policy, namely the behavior policy, and the target policy.

The *model-based* algorithms include *Value Iteration* and *Policy Iteration*, in which the model is known a prior to the user. In the field of *model-free* algorithms, there are *Monte-Carlo*, *Sarsa*, *Q-Learining(Deep Q-Learning)*, *Policy Gradient* and *A2C* with their variants. 

More advanced RL algorithms will be updated here in the future...

## How to practice with this repo?
All the algorithms introduced in the book are programmed in the folder ```Algorithms```. Run ```main.py``` which is in the folder ```examples``` to see how each of them works. You had better comment others before you test one.

Remeber to check your ```args``` in the script ```arguments.py``` and the environment tested in advance. Some crucial hyperparameters in the approximation methods are set in default and you are free to modify them.

## Plan for this repo in the future
More about the Multi-agent Reinforcement Learning will be posted here, including the implementation of relative algorithms as well as some brief description...

## Gratitude to Prof. Shiyu Zhao

None of these would be here without the great contribution made by Prof. Zhao. Here is the link to the homepage of this wonderful course: [To see a new world](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main).
