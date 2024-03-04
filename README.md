# AdaptiveFL: Communication-Adaptive Federated Learning under Dynamic Bandwidth 

Code for the following paper:
"AdaptiveFL: Communication-Adaptive Federated Learning under Dynamic Bandwidth".

## Introduction

Federated Learning (FL) is a distributed machine learning paradigm that enables heterogeneous devices to collaboratively train a model. Recognizing communication as a bottleneck in FL, existing communication-efficient solutions, e.g., HeteroFL and LotteryFL, etc, utilize gradient sparsification to reduce communication cost. However, existing solutions fail to address the \textbf{dynamic bandwidth} issue-- in which the bandwidth of each client is constantly changing throughout the training process. In this paper, we propose AdaptiveFL, a communication-adaptive FL framework, considering the dynamic constraints of bandwidth. The design of  AdaptiveFL follows two key steps. i) In each round,  each device selects a best-fit sub-model for communication per currently available bandwidth. ii) To guarantee the performance of each sub-model sent under dynamic bandwidth constraints, AdaptiveFL employs a local training method that enables each device to train a ``tailorable" local model, which can be tailored to any sparsity with competitive accuracy. We compare AdaptiveFL with several communication-efficient SOTA methods and demonstrate that AdaptiveFL outperforms other baselines by a large margin.

## Requirements
* Python 3.8
* PyTorch 1.13

## Usage

### Train ResNet9 on CIFAR10/CIFATR100/TinyImageNet:
Running ``sdfl_experiments/standalone/AdaptiveFL/adaptivefl_exp.py``.


### Parameters

All training/inference/model parameters are controlled from ``sdfl_experiments/standalone/AdaptiveFL/adaptivefl_exp.py``.
