import json
from shutil import copyfile

import numpy as np
import tensorflow as tf
from stable_baselines import TD3
from stable_baselines.gail import ExpertDataset
from stable_baselines.td3.policies import FeedForwardPolicy


from A2C_Train import NetworkParameters
from SocialNetwork import SN_Env


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

    loc_pretrain_dataset = "logs/pretraining_datasets/n30_8t_2f_5iter_0.02var.npz"
    # generate_pretraining_dataset(env, 2048000, loc_pretrain_dataset)

    copyfile("config.json", args.save_dir + "/config.json")


    class CustomTD3Policy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomTD3Policy, self).__init__(*args, **kwargs,
                                                  layers=[128, 128, 128, 128, 128, 64],
                                                  act_fun=tf.nn.tanh,
                                                  feature_extraction="mlp")


    model = TD3(CustomTD3Policy, env, verbose=1, tensorboard_log=args.save_dir + "/tensorboard_log/",
                learning_rate=0.000001, learning_starts=1000)

    dataset = ExpertDataset(expert_path=loc_pretrain_dataset, train_fraction=0.95, batch_size=128)
    model.pretrain(dataset, n_epochs=15, learning_rate=0.0002, val_interval=1, adam_epsilon=1e-04)
    model.save(args.save_dir + "/model_pretraining")
    del dataset

    model.learn(total_timesteps=3000000)
    model.save(args.save_dir + "/model_complete")
