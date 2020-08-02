# This is based on the record_expert.py file of stable baseline!

import numpy as np
import sys
from typing import Dict


def generate_pretraining_dataset(env, episodes, save_path):

    print("Create pretraining dataset...")
    old_disp_value = env.display_statistics
    env.display_statistics = False
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0

    while ep_idx < episodes:
        observations.append(obs)
        action = get_optimal_action(env)
        obs, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)

        reward_sum += reward
        idx += 1
        if done:
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1
            if ep_idx % 100 == 0:
                sys.stdout.write('\r')
                sys.stdout.write(str(ep_idx) + "/" + str(episodes) + " episodes recorded")
                sys.stdout.flush()

    observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    print("Saving dataset..")

    np.savez(save_path, **numpy_dict)

    env.display_statistics = old_disp_value

    print("\nRecording of pretraining dataset completed!\n")


def get_optimal_action(env):
    if env.excluded_members / env.numb_nodes <= 0.5:
        def is_true_value(x): return True if np.abs(x - env.sources_true_value) < 0.05 else False
        def is_false_value(x): return True if np.abs(x - env.sources_true_value) > 0.2 else False
        def is_excluded(x): return True if x == -1 else False
        def calc_opt_action(x): return 0.0 if (is_excluded(x) or is_false_value(x)) else 1.0 if is_true_value(x) else 0.5
        vf = np.vectorize(calc_opt_action)
        return vf(env.state)
    else:
        return np.ones((env.numb_nodes,))
