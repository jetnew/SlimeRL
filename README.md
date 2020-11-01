# Slime-RL

Code repository for the research project ["You Play Ball, I Play Ball: Bayesian Multi-Agent Reinforcement Learning for Slime Volleyball"](https://www.slime-rl.github.io). 

Presented at National University of Singapore [17th School of Computing Term Project Showcase](https://isteps.comp.nus.edu.sg/event/17th-steps/module/CS3244/project/4) (17th STePS).

<img src="https://user-images.githubusercontent.com/27071473/96207264-5ed17700-0f9d-11eb-80e5-8baee2408895.png">

## Summary

In [Slime Volleyball](https://github.com/hardmaru/slimevolleygym), a two-player competitive game, we investigate how <ins>learning under uncertainty</ins> improves AI players’ learning in 3 ways in the domain of multi-agent reinforcement learning (MARL):

1. Against an expert
2. Against itself
3. Against each other

We show that <ins>by modelling uncertainty</ins>, Bayesian methods do improve MARL training performance, and through experiments using TensorFlow Probability and Stable Baselines, we present interesting differences in agent behaviour.
