import base64
import IPython
import imageio

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

def record_game(model, env, num_episodes=5, video_filename='video.mp4'):
    with imageio.get_writer(video_filename, fps=60) as video:
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            video.append_data(env.render('rgb_array'))

            while not done:
                action, _steps = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                video.append_data(env.render('rgb_array'))

            print("score:", total_reward)
            
            
import numpy as np
import pandas as pd
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize


def evaluate(model, env, render=False):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
    return total_reward

def hyperopt(model, params, opt_params, trials=30):
    def loss_function(p):
        m = model(params['policy'],
                  params['train_env'],
                  **p)
        m.learn(total_timesteps=params['timesteps'])
        reward = np.mean([evaluate(m, params['eval_env']) for _ in range(100)])
        return {
            '-reward': (-reward, 0.0)}
    best_params, best_vals, experiment, exp_model = optimize(
        parameters=[{'name': name, 'type': 'range', 'bounds': bounds}
                    for name, bounds in opt_params.items()],
        evaluation_function=loss_function,
        objective_name="-reward",
        minimize=True,
        total_trials=trials)
    
    m = model(params['policy'],
              params['train_env'],
              **best_params)
    m.learn(total_timesteps=params['timesteps'],
                callback=params['eval_callback'])
    return m, best_params, best_vals, experiment, exp_model


def hyperopt_log(experiment):
    # Get parameter set for every trial
    df_experiment = pd.DataFrame([trial.arm.parameters for trial in experiment.trials.values()])
    
    # Get metrics for every trial
    df_metrics = experiment.fetch_data().df
    metric_names = df_metrics['metric_name'].unique()
    for metric_name in metric_names:
        metric_series = df_metrics[df_metrics['metric_name'] == metric_name]['mean'].reset_index(drop=True)
        df_experiment[metric_name] = metric_series
        
    return df_experiment


def hyperparam_plot(exp_model, param_x, param_y):
    render(plot_contour(exp_model, param_x, param_y, metric_name='-reward'))
    
def performance_plot(experiment, best_vals):
    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        optimum=best_vals[0]['-reward'],
        title="Model performance vs. # of iterations",
        ylabel="loss")
    render(best_objective_plot)
    
    
import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.ppo1 import PPO1
from stable_baselines import A2C, ACER, ACKTR, DQN, HER, GAIL, TRPO

import base64
import IPython
import imageio

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

def record_game(model, env, num_episodes=5, video_filename='video.mp4'):
    with imageio.get_writer(video_filename, fps=60) as video:
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            video.append_data(env.render('rgb_array'))

            while not done:
                action, _steps = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                video.append_data(env.render('rgb_array'))

            print("score:", total_reward)

def experiment(algo, policy, 
               seed=3244,
               timesteps=15_000_000,
               eval_freq=2_500,
               eval_episodes=100,
               record=False,
               tag='0'):
    """
    algo: Choose from
        a2c, acer, acktr, dqn, trpo, ppo1
    policy: Choose from
        mlp, bnn
    """
    if algo == "a2c":
        model = A2C
    elif algo == "acer":
        model = ACER
    elif algo == "dqn":
        model = DQN
    elif algo == "trpo":
        model = TRPO
    elif algo == "ppo":
        model = PPO1        
    
    if policy == "dnn":
        if algo == "dqn":
            from stable_baselines.deepq.policies import MlpPolicy
        else:
            from stable_baselines.common.policies import MlpPolicy
        policyFn = MlpPolicy
    elif policy == "bnn":
        if algo == "dqn":
            from dqn_model import BnnPolicy
        else:
            from model import BnnPolicy
        policyFn = BnnPolicy
        
        
    log_dir = f"{algo}-{policy}-{tag}"
    logger.configure(folder=log_dir)
    
    env = gym.make("SlimeVolley-v0")
    env.atari_mode = True
    env.survival_bonus = True
    env.__init__()
    env.seed(seed)
    
    eval_env = gym.make("SlimeVolley-v0")
    eval_env.atari_mode = True
    eval_env.__init__()
    eval_env.seed(seed)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 eval_freq=eval_freq,
                                 n_eval_episodes=eval_episodes)
    
    print(f"Beginning training for {algo}-{policy}-{tag}.")
    params = {
        'policy': policyFn,
        'train_env': env,
        'eval_env': eval_env,
        'timesteps': timesteps,
        'eval_callback': eval_callback,
    }
    opt_params = {
        'a2c': {
            'gamma': [0.900, 0.999],
            'vf_coef': [0.10, 0.40],
            'ent_coef': [0.001, 0.100],
            'max_grad_norm': [0.1, 0.9],
        },
        'acer': {
            'gamma': [0.900, 0.999],
            'q_coef': [0.1, 0.9],
            'ent_coef': [0.001, 0.100],
            'max_grad_norm': [1, 100],
        },
        'dqn': {
            'gamma': [0.900, 0.999],
            'exploration_fraction': [0.01, 0.20],
            'exploration_final_eps': [0.001, 0.030],
            'prioritized_replay_alpha': [0.3, 0.9]
        },
        'trpo': {
            'gamma': [0.900, 0.999],
            'max_kl': [0.001, 0.100],
            'lam': [0.90, 0.99],
            'entcoeff': [0.001, 0.100],
            'cg_damping': [0.001, 0.100],
            'vf_stepsize': [0.00001, 0.00100],
        },
        'ppo': {
            'clip_param': [0.01, 0.99],
            'entcoeff': [0.001, 0.100],
            'gamma': [0.900, 0.999],
            'lam': [0.90, 0.99]
        },
    }
    
    model, best_params, best_vals, experiment, exp_model = hyperopt(model, params, opt_params[algo])
    with open(os.path.join(log_dir, "best_params.txt"), 'w') as f:
        f.write(str(best_params))

    model.save(os.path.join(log_dir, "trained_model"))
    print("Training complete, saved to:", os.path.join(log_dir, f"trained-{algo}-{policy}-{tag}"))
    
    if record:
        print("Recording video of gameplay.")
        video_filename=os.path.join(log_dir, f"trained-{algo}-{policy}-{tag}.mp4")
        record_game(model=model,
                    env=env,
                    num_episodes=5,
                    video_filename=video_filename)
    
    env.close()
    return model, best_params, best_vals, experiment, exp_model
    
    
    
if __name__ == "__main__":
    for i in range(20):
        # =========================================
        # HERE IS WHERE YOU CHANGE THE CODE
        # 1. Change 'ppo' to the algorithm. [a2c, acer, acktr, dqn, gail, trpo, ppo]
        # 2. Change 'dnn' to the policy. [dnn, bnn]
        experiment('ppo', 'dnn', timesteps=5_000_000, record=True, tag=str(i))
        # =========================================