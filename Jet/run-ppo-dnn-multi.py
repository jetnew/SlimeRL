import os
import time
import gym
import slimevolleygym
import numpy as np

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor

from shutil import copyfile # keep track of generations


class SlimeVolleyMultiAgentEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best previous opponent model
  def __init__(self):
    super(SlimeVolleyMultiAgentEnv, self).__init__()
    self.policy = self
    self.opp_model = None
    self.opp_model_filename = None
    self.self_model_gen = 0
    self.opp_model_gen = 0
  def predict(self, obs): # the policy
    if self.opp_model is None:
      return self.action_space.sample() # return a random action
    else:
      action, _ = self.opp_model.predict(obs)
      return action
    
  def reset(self):
    # load model if it's there, else wait (meant to be used in parallel with opponent training)
    os.makedirs(OPP_LOGDIR, exist_ok=True)
    opp_modellist = [f for f in os.listdir(OPP_LOGDIR) if f.startswith("history")]
    opp_modellist.sort()
    
    os.makedirs(SELF_LOGDIR, exist_ok=True)
    self_modellist = [f for f in os.listdir(SELF_LOGDIR) if f.startswith("history")]
    self_modellist.sort()
    
    while True:
        os.makedirs(OPP_LOGDIR, exist_ok=True)
        opp_modellist = [f for f in os.listdir(OPP_LOGDIR) if f.startswith("history")]
        opp_modellist.sort()
        
        os.makedirs(SELF_LOGDIR, exist_ok=True)
        self_modellist = [f for f in os.listdir(SELF_LOGDIR) if f.startswith("history")]
        self_modellist.sort()
    
        if len(self_modellist) == 0:
            print("debug2:", len(self_modellist))
            return super(SlimeVolleyMultiAgentEnv, self).reset()
        elif 0 <= len(opp_modellist) - len(self_modellist) <= 1:
            print("debug3:", len(self_modellist), len(opp_modellist))
            opp_filename = opp_modellist[-1]
            self.opp_model = PPO1.load(os.path.join(OPP_LOGDIR, opp_filename), env=self)
            return super(SlimeVolleyMultiAgentEnv, self).reset()
        print("Waiting for opponent training to complete.")
        time.sleep(5)
            

class MultiAgentCallback(EvalCallback):
  # hacked it to save new version of best model after TIMESTEPS_PER_GEN timesteps
  def __init__(self, *args, **kwargs):
    super(MultiAgentCallback, self).__init__(*args, **kwargs)
    self.generation = 1
  def _on_step(self) -> bool:
    result = super(MultiAgentCallback, self)._on_step()
    if result and self.num_timesteps > self.generation * TIMESTEPS_PER_GEN:
      print("MULTIAGENT: updating generation to", self.generation)
      
      self.model.save(os.path.join(SELF_LOGDIR, "history_"+str(self.generation).zfill(8)+".zip"))
      self.generation += 1
    return result


if __name__=="__main__":
  SEED = 0
  NUM_TIMESTEPS = 50_000_000
  TIMESTEPS_PER_GEN = 1000#100_000
  EVAL_FREQ = 1000#100_000
  EVAL_EPISODES = 1#100
  
  RENDER_MODE = False

  SELF_LOGDIR = "exp/multi/ppo-dnn"
  OPP_LOGDIR = "exp/multi/ppo-bnn"
  logger.configure(folder=SELF_LOGDIR)

  env = SlimeVolleyMultiAgentEnv()
  env = Monitor(env, SELF_LOGDIR, allow_early_resets=True)
  env.seed(SEED)

  # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
  model = PPO1(MlpPolicy, env, verbose=2)

  eval_callback = MultiAgentCallback(env,
    best_model_save_path=SELF_LOGDIR,
    log_path=SELF_LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(SELF_LOGDIR, "final_model"))

  env.close()
