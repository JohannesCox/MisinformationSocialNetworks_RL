import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from stable_baselines import A2C, TRPO, PPO1
from SocialNetwork import SN_Env
from A2C_Train import NetworkParameters

save_dir = "logs/A2C_9t_1f_rf3/"
play_iterations = 300
random_exclude = 0
gamma = 0.95
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[48, 16])
model = A2C.load(save_dir + "model", policy_kwargs=policy_kwargs)


def read_in_parameters(file):
    with open(file + "config.json") as json_file:
        data = json.load(json_file)
        return NetworkParameters(data)

args = read_in_parameters(save_dir)

class RandomAgent:
    def __init__(self, random_exclude_ra):
        self.random_exclude = random_exclude

    def play(self, env):
        reward_random = 0
        all_discounted_rewards_random = np.zeros((args.max_iterations,))
        done = False
        i = 0
        while not done:
            action_random = np.zeros(args.numb_nodes, )
            action_random[:self.random_exclude] = 1
            np.random.shuffle(action_random)
            _, reward, done, _ = env.step(action_random)
            reward_random = reward_random + reward * gamma ** i
            all_discounted_rewards_random[i] = reward_random
            i += 1
        return reward_random, all_discounted_rewards_random

def compare_model_random_agent():
    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false, args.max_iterations, playing=True)

    random_agent = RandomAgent(0)
    episode_reward_all_model = np.zeros((play_iterations, args.max_iterations))
    episode_reward_all_RA = np.zeros((play_iterations, args.max_iterations))
    all_discounted_rewards_random = np.zeros((args.max_iterations, play_iterations))
    all_discounted_rewards_model = np.zeros((args.max_iterations, play_iterations))

    seed_random = np.random.randint(1,10000)

    for i in range(play_iterations):
        # Play with model
        obs = env.reset(True, i * seed_random)
        episode_reward_model = 0
        discounted_rewards_model = np.zeros(args.max_iterations,)
        iteration = 0
        dones = False
        while (not dones):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            episode_reward_model = episode_reward_model + rewards * gamma ** iteration
            discounted_rewards_model[iteration] = episode_reward_model
            iteration += 1
        excluded_members = env.excluded_members
        # Play with random agent
        env.reset(True, i * seed_random)
        episode_reward_randomAgent, discounted_rewards_random = random_agent.play(env)

        print("Iteration " + str(i + 1) + "/" + str(play_iterations) + " | Episode reward: " + str(episode_reward_model) + " | " + "steps: " +
              str(iteration) + " | " + "excluded members: " + str(int(excluded_members)) + " | Episode reward RA: " + str(episode_reward_randomAgent))
        episode_reward_all_model[i] = episode_reward_model
        episode_reward_all_RA[i] = episode_reward_randomAgent
        all_discounted_rewards_random[:,i] = discounted_rewards_random
        all_discounted_rewards_model[:,i] = discounted_rewards_model

    print("Average reward model: " + str(np.mean(episode_reward_all_model)))
    print("Average reward random agent: " + str(np.mean(episode_reward_all_RA)))
    make_figure_rewardIteration(np.mean(all_discounted_rewards_random, axis=1), np.mean(all_discounted_rewards_model, axis=1), "random_agent", "model")

def make_figure_rewardIteration(rewards1, rewards2, label1, label2, iterations=args.max_iterations, gamma=0.95):
    optimum = np.zeros((iterations,))
    reward = 0
    for i in range(iterations):
        reward = reward + 1 * gamma ** i
        optimum[i] = reward
    plt.figure()
    plt.plot(range(iterations), rewards1, label=label1)
    plt.plot(range(iterations), rewards2, label=label2)
    plt.plot(range(iterations), optimum, label="theoretical optimum")
    plt.xlabel("iteration")
    plt.ylabel("discounted reward")
    plt.legend()
    plt.show()
    plt.savefig(args.save_dir + "/reward_per_iteration")



def compare_two_random_agents(exclude1, exclude2):
    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false,
                 args.max_iterations, playing=True)

    random_agent_1 = RandomAgent(exclude1)
    random_agent_2 = RandomAgent(exclude2)
    episode_reward_all_RA1 = np.zeros((play_iterations, 1))
    episode_reward_all_RA2 = np.zeros((play_iterations, 1))

    for i in range(play_iterations):
        # Play with random agent 1
        env.reset(True, i)
        episode_reward_randomAgent1 = random_agent_1.play(env)

        # Play with random agent 2
        env.reset(True, i)
        episode_reward_randomAgent2 = random_agent_2.play(env)

        print("Iteration " + str(i + 1) + "/" + str(play_iterations) + " | Episode reward RA1: " + str(episode_reward_randomAgent1) + " | " + "Episode reward RA2: " +
              str(episode_reward_randomAgent2))
        episode_reward_all_RA1[i] = episode_reward_randomAgent1
        episode_reward_all_RA2[i] = episode_reward_randomAgent2

    print("Average reward random agent 1: " + str(np.mean(episode_reward_all_RA1)))
    print("Average reward random agent 2: " + str(np.mean(episode_reward_all_RA2)))

if __name__ == '__main__':
    compare_model_random_agent()
    #compare_two_random_agents(0,1)