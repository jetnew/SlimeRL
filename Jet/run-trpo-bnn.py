import os
import numpy as np
import pandas as pd

import gym
import slimevolleygym

from stable_baselines import TRPO
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

from model import BnnPolicy


NUM_TIMESTEPS = 5000#5_000_000
EVAL_FREQ = 100_000
EVAL_EPISODES = 1_000
NUM_TRIALS = 10


for n in range(1, NUM_TRIALS + 1):
    LOGDIR = f"exp/trpo-bnn/{n}"
    logger.configure(folder=LOGDIR)

    env = gym.make("SlimeVolley-v0")
    env.atari_mode = True
    env.__init__()
    env.seed(n)


    model = TRPO(BnnPolicy, env, verbose=2)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model"))

    env.close()