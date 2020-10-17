import os
import numpy as np
import pandas as pd

import gym
import slimevolleygym

from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor


NUM_TIMESTEPS = 5_000_000
EVAL_FREQ = 100_000
EVAL_EPISODES = 1_000
NUM_TRIALS = 10


for n in range(1, NUM_TRIALS + 1):
    LOGDIR = f"exp/expert/a2c-dnn/{n}"
    logger.configure(folder=LOGDIR)

    env = gym.make("SlimeVolley-v0")
    env = Monitor(env, LOGDIR, allow_early_resets=True)
    env.seed(n)


    model = A2C(MlpPolicy, env, verbose=2)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR,         eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model"))

    env.close()