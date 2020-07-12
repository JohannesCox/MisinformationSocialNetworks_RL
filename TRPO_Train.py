import json
from shutil import copyfile

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

from SocialNetwork import SN_Env

class NetworkParameters:
  def __init__(self, data):
      self.numb_nodes = data["numb_nodes"] #1000
      self.connectivity = data["connectivity"] #0.1
      self.numb_sources_true = data["numb_sources_true"] #4
      self.numb_sources_false = data["numb_sources_false"] #6
      self.max_iterations = data["max_iterations"] #100
      self.max_reward = data["max_reward"] #100

      self.save_dir = data["save_dir"]

def read_in_parameters():
    with open('config.json') as json_file:
      data = json.load(json_file)
      return NetworkParameters(data)

if __name__ == '__main__':

    args = read_in_parameters()
    env = SN_Env(args.numb_nodes, args.connectivity, args.numb_sources_true, args.numb_sources_false,
                 args.max_iterations, True, 1000000, 2000000)

    if not np.os.path.exists(args.save_dir):
        np.os.mkdir(args.save_dir)

    model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log=args.save_dir + "/tensorboard_log/")
    model.learn(total_timesteps=10000000)
    model.save(args.save_dir + "/model")
    copyfile("config.json", args.save_dir + "/config.json")