from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from aviary import Aviary
import torch

if __name__ == '__main__':
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
                      act=ACTION_TYPE)

    POLICY_ARGS = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=torch.nn.Tanh
    )

    model = A2C('MlpPolicy', train_env, policy_kwargs=POLICY_ARGS)

    REWARD_THRESHOLD = 467

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD,
                                                     verbose=True)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=True,
                                 best_model_save_path='best_models/',
                                 log_path='logs/',
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=True)

    TOTAL_TIMESTEPS = int(1e7)

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                log_interval=100)

    model.save('final_model.zip')
