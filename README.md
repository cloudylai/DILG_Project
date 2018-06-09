# DILG Project
Deep imitation learning game project

## Introduction
In this project, we attempt to improve the performance and training efficiency of imitation learning. In imitation learning, one attempt to train an agent to behave like an expert without any predefined reward function. What the agent can do is to learn its policy by imitating the demonstrative trajectories from the expert. Specifically, we study imitation learning and extend the generative adversarial imitation learning framework (GAIL) by Jonathan Ho and Stefano Ermon[[link](https://arxiv.org/abs/1606.03476)]. Our model is based on some observations and potential issues about this learning framework. We conduct our designed model on several environments in OpenAI Gym[[link](https://gym.openai.com/), and implement our model based on the repository[[link](https://github.com/openai/imitation)].

## Method
We briefly describe our idea behind our designed model. In GAIL framework, a policy network is used to make actions which are similar to the demonstration, and a critic network is used to discriminate the actions between the policy network and the demonstrative samples from the expert. Adversarial training is applied to these two networks. Our main observation is that the critic network in GAIL discriminates an action only based on the state-action pair in the last step. In order to better the discrimination, we suggest that the critic network should judge an action based on multi-step state-action pairs. Following this idea, we attempt sequential adversarial imitation learning (SAIL) and replace the feed-forward network with Long Short-Term Memory (LSTM) network as the critic network. The below figure is a simple diagram to show the difference between GAIL and SAIL. To train our model, we follow the same strategy as GAIL, where we use trust region policy optimization (TRPO)[[link](https://arxiv.org/abs/1502.05477)] with a value network to update the policy network and train the critic network as a classifier to minimize the classification loss between generated actions and demonstrative actions. 

![image1](https://github.com/cloudylai/DILG_Project/blob/master/images/digram_1.png) 

## Result
We compare our model (SAIL) with GAIL and behavior cloning (BC), a baseline model which is a supervised classifier and predicts the next action based on the current state-action pair. We conduct these models on two environments: Cartpole-v0 and MountainCar-v0 in OpenAI Gym. The following demonstrates the reulsts. It turn out that SAIL can achieve similar results as GAIL when the training data is large enough but with higher variance. We suppose three possible reasons that SAIL does not outperform GAIL: (1) Training LSTM is more difficult than training feed-forward network. (2) The testing environments are simple  (3) Judging a sequence of state-action pairs is a very difficult task for critic network. 

![image2](https://github.com/cloudylai/DILG_Project/blob/master/results/rel_sga13_cartpole_r2.png) 

![image3](https://github.com/cloudylai/DILG_Project/blob/master/results/rel_sga13_mountain_r0.png) 

## Reference
### Paper:  
1. Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. NIPS 2016. [link](https://arxiv.org/abs/1606.03476)  
2. John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel. Trust Region Policy Optimization. ICML 2015. [link](https://arxiv.org/abs/1502.05477)
### github:  
1. openai imiation: [link](https://github.com/openai/imitation) 
