import os
import time
import gym
import slimevolleygym
import numpy as np

from stable_baselines.ppo1 import PPO1
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor
from model import BnnPolicy

from shutil import copyfile # keep track of generations


class SlimeVolleyMultiAgentEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best previous opponent model
  def __init__(self):
    super(SlimeVolleyMultiAgentEnv, self).__init__()
    self.policy = self
    self.opp_model = None
    self.opp_model_filename = None
  def predict(self, obs): # the policy
    if self.opp_model is None:
      return self.action_space.sample() # return a random action
    else:
      action, _ = self.opp_model.predict(obs)
      return action
    
  def reset(self):
    # Load model if it's there, else wait (meant to be used in parallel with opponent training)
    # reset() is run multiple times throughout the experiment, not just during callbacks.
    
    while True:
        opp_modellist = [f for f in os.listdir(OPP_LOGDIR) if f.startswith("history")]
        opp_modellist.sort()
        
        self_modellist = [f for f in os.listdir(SELF_LOGDIR) if f.startswith("history")]
        self_modellist.sort()
        
        # Experiment just started, so no history files
        if len(self_modellist) == 0:
            return super(SlimeVolleyMultiAgentEnv, self).reset()
        
        # Middle of experiment
        if len(self_modellist) > 0:
            # If num of history files is the same, check opponent's last gen.
            if len(self_modellist) - len(opp_modellist) == 0:
                opp_filename = opp_modellist[-1]
                # Opponent's last gen has no change -> Both models still training the same gen
                if opp_filename == self.opp_model_filename:
                    return super(SlimeVolleyMultiAgentEnv, self).reset()
                # Opponent's last gen changed -> Opponent model has been waiting -> Load new opp.
                elif opp_filename != self.opp_model_filename:
                    print("Loading model:", opp_filename)
                    self.opp_model_filename = opp_filename
                    if self.opp_model is not None:
                        del self.opp_model
                    self.opp_model = PPO1.load(os.path.join(OPP_LOGDIR, opp_filename), env=self)
                    return super(SlimeVolleyMultiAgentEnv, self).reset()
            # Opponent's finished current gen training, self should continue training.
            elif len(opp_modellist) - len(self_modellist) == 1:
                print(f"Self: Gen {len(self_modellist)}, Opp: Gen {len(opp_modellist)}. Opponent waiting for self training to complete.")
                return super(SlimeVolleyMultiAgentEnv, self).reset()
        print(f"Self: Gen {len(self_modellist)}, Opp: Gen {len(opp_modellist)}. Waiting for opponent training to complete.")
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
  TIMESTEPS_PER_GEN = 100_000
  EVAL_FREQ = 100_000
  EVAL_EPISODES = 100
  
  RENDER_MODE = False

  SELF_LOGDIR = "exp/multi/ppo-bnn-mujoco"
  OPP_LOGDIR = "exp/multi/ppo-dnn-mujoco"
  logger.configure(folder=SELF_LOGDIR)

  env = SlimeVolleyMultiAgentEnv()
  env.seed(SEED)

  # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
  model = PPO1(BnnPolicy, env,
        timesteps_per_actorbatch=4096,
        clip_param=0.2,
        entcoeff=0.0,
        optim_epochs=10,
        optim_stepsize=3e-4,
        optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear', verbose=2)

  eval_callback = MultiAgentCallback(env,
    best_model_save_path=SELF_LOGDIR,
    log_path=SELF_LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(SELF_LOGDIR, "final_model"))

  env.close()
