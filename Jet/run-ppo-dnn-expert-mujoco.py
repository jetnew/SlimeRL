import os
import numpy as np
import pandas as pd

import gym
import slimevolleygym

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor


NUM_TIMESTEPS = 5_000_000
EVAL_FREQ = 100_000
EVAL_EPISODES = 100
NUM_TRIALS = 7


for n in range(4, NUM_TRIALS + 1):
    LOGDIR = f"exp/expert/ppo-dnn-mujoco/{n}"
    logger.configure(folder=LOGDIR)

    env = gym.make("SlimeVolley-v0")
    env = Monitor(env, LOGDIR, allow_early_resets=True)
    env.seed(n)


    model = PPO1(MlpPolicy, env,
        timesteps_per_actorbatch=4096,
        clip_param=0.2,
        entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=3e-4,
        optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear', verbose=2)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model"))

    env.close()
