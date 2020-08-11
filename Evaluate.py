import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from stable_baselines import A2C, PPO1

from A2C_Train import NetworkParameters
from Create_Pretraining_Dataset import get_optimal_action
from SocialNetwork import SN_Env

save_dir = "logs/A2C_9t_1f_100nodes_MLP_10iter_v2/"
play_iterations = 300
gamma = 0.99
policy_kwargs = dict(net_arch=[1024, 1024, 1024, dict(vf=[1024, 512], pi=[1024, 1024, 512])], act_fun=tf.nn.tanh)
model_complete = A2C.load(save_dir + "model_complete", policy_kwargs=policy_kwargs)
model_pretraining = A2C.load(save_dir + "model_pretraining", policy_kwargs=policy_kwargs)


def read_in_parameters(file):
    with open(file + "config.json") as json_file:
        data = json.load(json_file)
        return NetworkParameters(data)


args = read_in_parameters(save_dir)


class DoNothingAgent:
    def play(self, env):
        reward_random = 0
        all_discounted_rewards_random = np.zeros((args.max_iterations,))
        done = False
        i = 0
        while not done:
            action = np.ones(args.numb_nodes, )
            _, reward, done, _ = env.step(action)
            reward_random = reward_random + reward * gamma ** i
            all_discounted_rewards_random[i] = reward_random
            i += 1
        return reward_random, all_discounted_rewards_random


def compare_model_random_agent():
    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false,
                 args.max_iterations, playing=True, display_statistics=False,
                 initial_noise_variance=args.initial_noise_variance)

    random_agent = DoNothingAgent()
    episode_reward_all_model_complete = np.zeros((play_iterations, args.max_iterations))
    episode_reward_all_model_pretraining = np.zeros((play_iterations, args.max_iterations))
    episode_reward_all_dn_agent = np.zeros((play_iterations, args.max_iterations))
    all_discounted_rewards_dn_agent = np.zeros((args.max_iterations, play_iterations))
    all_discounted_rewards_model_complete = np.zeros((args.max_iterations, play_iterations))
    all_discounted_rewards_model_pretraining = np.zeros((args.max_iterations, play_iterations))

    seed_random = np.random.randint(1, 1000)

    for i in range(play_iterations):
        # Play with model complete
        obs = env.reset(True, i * seed_random)
        episode_reward_model_complete = 0
        discounted_rewards_model_complete = np.zeros(args.max_iterations, )
        iteration = 0
        done = False
        while not done:
            action, _states = model_complete.predict(obs)
            obs, rewards, done, info = env.step(action)
            episode_reward_model_complete = episode_reward_model_complete + rewards * gamma ** iteration
            discounted_rewards_model_complete[iteration] = episode_reward_model_complete
            iteration += 1
        excluded_members_complete_model = env.excluded_members

        # Play with pretraining only agent
        obs = env.reset(True, i * seed_random)
        episode_reward_model_pretraining = 0
        discounted_rewards_model_pretraining = np.zeros(args.max_iterations, )
        iteration = 0
        done = False
        while not done:
            action, _states = model_pretraining.predict(obs)
            obs, rewards, done, info = env.step(action)
            episode_reward_model_pretraining = episode_reward_model_pretraining + rewards * gamma ** iteration
            discounted_rewards_model_pretraining[iteration] = episode_reward_model_pretraining
            iteration += 1
        excluded_members_pretraining_model = env.excluded_members

        # Play with do nothing agent
        env.reset(True, i * seed_random)
        episode_reward_dn_agent, discounted_rewards_random = random_agent.play(env)

        print("Iteration " + str(i + 1) + "/" + str(play_iterations) + " | Episode reward cm: " + str(
            episode_reward_model_complete) + " | " + "steps cm: " +
              str(iteration) + " | " + "excluded members cm: " + str(
            int(excluded_members_complete_model)) + " | Episode reward ptm: " + str(
            episode_reward_model_pretraining) + " | " + " | " + "excluded members ptm: " + str(
            int(excluded_members_pretraining_model)) + " | Episode reward dn agent: " + str(episode_reward_dn_agent))
        episode_reward_all_model_complete[i] = episode_reward_model_complete
        episode_reward_all_model_pretraining[i] = episode_reward_model_pretraining
        episode_reward_all_dn_agent[i] = episode_reward_dn_agent
        all_discounted_rewards_dn_agent[:, i] = discounted_rewards_random
        all_discounted_rewards_model_complete[:, i] = discounted_rewards_model_complete
        all_discounted_rewards_model_pretraining[:, i] = discounted_rewards_model_pretraining

    print("Average reward complete model: " + str(np.mean(episode_reward_all_model_complete)))
    print("Average reward pretraining only model: " + str(np.mean(episode_reward_all_model_pretraining)))
    print("Average reward do nothing agent: " + str(np.mean(episode_reward_all_dn_agent)))

    make_figure_reward_iteration(np.mean(all_discounted_rewards_dn_agent, axis=1),
                                 np.mean(all_discounted_rewards_model_complete, axis=1),
                                 np.mean(all_discounted_rewards_model_pretraining, axis=1), "random_agent",
                                 "model_complete", "model_pretraaining_only")


def make_figure_reward_iteration(rewards1, rewards2, rewards3, label1, label2, label3,
                                 iterations=args.max_iterations, ):
    optimum = np.zeros((iterations,))
    reward = 0
    for i in range(iterations):
        reward = reward + 1 * gamma ** i
        optimum[i] = reward
    plt.figure()
    plt.plot(range(iterations), rewards1, label=label1)
    plt.plot(range(iterations), rewards2, label=label2)
    plt.plot(range(iterations), rewards3, label=label3)
    # plt.plot(range(iterations), optimum, label="theoretical optimum")
    plt.xlabel("iteration")
    plt.ylabel("discounted reward")
    plt.legend()
    plt.savefig(args.save_dir + "/reward_per_iteration.png")
    plt.show()

def analize_pretrained_agent():
    initial_noise_variance = 0.02

    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false,
                 args.max_iterations, playing=True, display_statistics=False,
                 initial_noise_variance=args.initial_noise_variance)

    discounted_rewards_model_pretraining = np.zeros(play_iterations, )
    mses = np.zeros(play_iterations, )

    for i in range(play_iterations):
        # Play with pretraining only agent
        iteration = 0
        episode_reward_model_pretraining = 0
        done = False
        obs = env.reset()
        mse = 0
        while not done:
            action, _states = model_pretraining.predict(obs)
            action_opt = get_optimal_action(env)
            mse += (np.square(action - action_opt)).mean(axis=None)
            obs, rewards, done, info = env.step(action)
            episode_reward_model_pretraining = episode_reward_model_pretraining + rewards * gamma ** iteration
            iteration += 1
        mse = mse / iteration
        mses[i] = mse
        discounted_rewards_model_pretraining[i] = episode_reward_model_pretraining
        print("Iteration " + str(i + 1) + "/" + str(play_iterations) + " | MSE: " + str(
            mse) + " | reward pretrained model: " + str(
            episode_reward_model_pretraining) + " | excluded members: " + str(env.excluded_members))
    print("Done! Final statistics:")
    print("Average mse: " + str(np.mean(mses)) + " | average discounted rewards: " + str(
        np.mean(discounted_rewards_model_pretraining)))


if __name__ == '__main__':
    compare_model_random_agent()
    # analize_pretrained_agent()
