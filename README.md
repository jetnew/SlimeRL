# SlimeRL

Code repository for the research project ["You Play Ball, I Play Ball: Bayesian Multi-Agent Reinforcement Learning for Slime Volleyball"](https://www.slime-rl.github.io). 

Presented at National University of Singapore - [17th School of Computing Term Project Showcase](https://isteps.comp.nus.edu.sg/event/17th-steps/module/CS3244/project/3) (17th STePS).

<img src="https://user-images.githubusercontent.com/27071473/96207264-5ed17700-0f9d-11eb-80e5-8baee2408895.png">

## Demo & Video

Click [here](https://www.slimerl.tech) to view interesting agent behaviour and notice the differences between agents and their Bayesian counterparts! Click [here](https://www.youtube.com/watch?v=8qjV19gkZXc) for a introductory video.

## About

In [Slime Volleyball](https://github.com/hardmaru/slimevolleygym), a two-player competitive game, we investigate how <ins>modelling uncertainty</ins> improves AI players’ learning in 3 ways: 1) <ins>against an expert</ins>, 2) <ins>against itself</ins> and 3) <ins>against each other</ins>, in the domain of multi-agent reinforcement learning (MARL).

We show that by modelling uncertainty, Bayesian methods improve MARL training in 4 ways: 1) <ins>performance</ins>, 2) <ins>training stability</ins>, 3) <ins>uncertainty</ins> and 4) <ins>generalisability</ins>, and through experiments using [TensorFlow Probability](https://www.tensorflow.org/probability/) and [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/), we present interesting differences in agent behaviour.

We contribute code for 3 functionalities: 1) <ins>Bayesian methods using Flipout</ins> integrated into Stable Baselines, 2) <ins>Multi-agent versioned learning framework</ins> for Stable Baselines (previously with only single-agent support) and 3) <ins>Uncertainty visualisation using agent clones</ins> for Slime Volleyball Gym.
