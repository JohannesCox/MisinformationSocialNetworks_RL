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
        self.numb_nodes = data["numb_nodes"]
        self.connectivity = data["connectivity"]
        self.numb_sources_true = data["numb_sources_true"]
        self.numb_sources_false = data["numb_sources_false"]
        self.max_iterations = data["max_iterations"]
        self.initial_noise_variance = data["initial_noise_variance"]
        self.save_dir = data["save_dir"]


def read_in_parameters():
    with open('config.json') as json_file:
        data = json.load(json_file)
        return NetworkParameters(data)


if __name__ == '__main__':

    args = read_in_parameters()

    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false,
                 args.max_iterations, playing=False, initial_noise_variance=args.initial_noise_variance,
                 training_statistics_interval=1000)

    if not np.os.path.exists(args.save_dir):
        np.os.mkdir(args.save_dir)

    loc_pretrain_dataset = "logs/pretraining_datasets/n100_1t_9f_4iter_0.02var.npz"
    generate_pretraining_dataset(env, 600000, loc_pretrain_dataset)

    copyfile("config.json", args.save_dir + "/config.json")

    pk = dict(net_arch=[1024, 1024, 1024, dict(vf=[1024, 512], pi=[1024, 1024, 512])], act_fun=tf.nn.tanh)

    model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=args.save_dir + "/tensorboard_log/", ent_coef=0.04,
                gamma=0.99, learning_rate=0.000002, n_steps=20, epsilon=1e-04,
                policy_kwargs=pk)

    dataset = ExpertDataset(expert_path=loc_pretrain_dataset, train_fraction=0.95, batch_size=128)
    model.pretrain(dataset, n_epochs=20, learning_rate=0.0005, val_interval=1, adam_epsilon=1e-04)
    model.save(args.save_dir + "/model_pretraining")
    del dataset

    model.learn(total_timesteps=5000000)
    model.save(args.save_dir + "/model_complete")

