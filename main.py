from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import argparse

from aviary import Aviary
import torch

POLICIES = {
    'mlp_64_64': dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=torch.nn.Tanh),
    'mlp_128_128': dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])], activation_fn=torch.nn.Tanh),
    'mlp_32_32': dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])], activation_fn=torch.nn.Tanh),
    'mlp_64_64_64': dict(net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])], activation_fn=torch.nn.Tanh)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--policy_name', action='store', required=True)
    parser.add_argument('--run_name', action='store', required=True)
    parser.add_argument('--timesteps', action='store', required=True)

    args = parser.parse_args()

    OBSERVATION_TYPE = ObservationType('kin')
    ACTION_TYPE = ActionType('rpm')
    train_env = make_vec_env(Aviary,
                             n_envs=1,
                             env_kwargs=dict(
                                 obs=OBSERVATION_TYPE,
                                 act=ACTION_TYPE,
                                 gui=False
                             ))

    eval_env = Aviary(obs=OBSERVATION_TYPE,
                      act=ACTION_TYPE, gui=True)

    model = PPO('MlpPolicy', train_env, verbose=1, policy_kwargs=POLICIES[args.policy_name])
    # model = PPO.load('best_models_1/best_model.zip', train_env)
    REWARD_THRESHOLD = 1000

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD,
                                                     verbose=False)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=True,
                                 best_model_save_path=f'best_models/{args.run_name}/{args.policy_name}/',
                                 log_path=f'logs/{args.run_name}/{args.policy_name}/',
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=True,
                                 n_eval_episodes=5)

    model.learn(total_timesteps=int(args.timesteps),
                callback=eval_callback,
                log_interval=100)

    model.save(f'final_models/{args.run_name}/{args.policy_name}.zip')
