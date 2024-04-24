import time

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from aviary import Aviary
import torch

if __name__ == '__main__':
    model = A2C.load('final_model.zip')
    OBSERVATION_TYPE = ObservationType('kin')
    ACTION_TYPE = ActionType('rpm')
    test_env = Aviary(gui=True,
                      obs=OBSERVATION_TYPE,
                      act=ACTION_TYPE,
                      record=False)

    test_env_nogui = Aviary(obs=OBSERVATION_TYPE,
                            act=ACTION_TYPE)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder='logs/',
                    colab=False
                    )
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
              truncated)
        if OBSERVATION_TYPE == ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i / test_env.CTRL_FREQ,
                       state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                       control=np.zeros(12)
                           )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if OBSERVATION_TYPE == ObservationType.KIN:
        logger.plot()

