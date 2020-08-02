import json
from shutil import copyfile

import numpy as np
import tensorflow as tf
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail import ExpertDataset

from Create_Pretraining_Dataset import generate_pretraining_dataset
from SocialNetwork import SN_Env


class NetworkParameters:
    def __init__(self, data):
        self.numb_nodes = data["numb_nodes"]  # 1000
        self.connectivity = data["connectivity"]  # 0.1
        self.numb_sources_true = data["numb_sources_true"]  # 4
        self.numb_sources_false = data["numb_sources_false"]  # 6
        self.max_iterations = data["max_iterations"]  # 100

        self.save_dir = data["save_dir"]


def read_in_parameters():
    with open('config.json') as json_file:
        data = json.load(json_file)
        return NetworkParameters(data)


if __name__ == '__main__':

    args = read_in_parameters()
    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false,
                 args.max_iterations, playing=False)

    if not np.os.path.exists(args.save_dir):
        np.os.mkdir(args.save_dir)

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=args.save_dir + "/tensorboard_log/", ent_coef=0.02,
                gamma=0.99, policy_kwargs=policy_kwargs)

    generate_pretraining_dataset(env, 25600, "logs/pretraining_datasets/v50.1.npz")
    dataset = ExpertDataset(expert_path="logs/pretraining_datasets/v50.1.npz", train_fraction=0.95)

    model.pretrain(dataset, n_epochs=500)
    model.save(args.save_dir + "/model_pretraining")

    model.learn(total_timesteps=1000000)
    model.save(args.save_dir + "/model_complete")

    copyfile("config.json", args.save_dir + "/config.json")
