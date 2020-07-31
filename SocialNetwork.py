import gym
import numpy as np

from gym import spaces


class Statistics:
    def __init__(self):
        self.iteration = 0
        self.moving_avg_exclusions = 0
        self.ep_avg_exclusion = 0
        self.ep_min_exclusions = 0
        self.ep_max_exclusions = 0
        self.ep_variance_exclusions = 0

    def update_statistics(self, iteration_interval, final_exclusion_numbers):
        if self.iteration == 0:
            self.moving_avg_exclusions = np.mean(final_exclusion_numbers)
            self.iteration = iteration_interval
        else:
            self.moving_avg_exclusions = self.moving_avg_exclusions * 0.9 + 0.1 * np.mean(final_exclusion_numbers)
            self.iteration += iteration_interval

        self.ep_avg_exclusion = np.mean(final_exclusion_numbers)
        self.ep_min_exclusions = np.min(final_exclusion_numbers)
        self.ep_max_exclusions = np.max(final_exclusion_numbers)
        self.ep_variance_exclusions = np.var(final_exclusion_numbers)

    def print_statistics(self):
        print("")
        print("")
        print("-------------------------------")
        print("Exclusion Statistics:")
        print("Total iterations " + str(self.iteration) + " | " + "Moving avg exclusions: " + str(
            self.moving_avg_exclusions) + " | " + "Ep exclusions avg: " + str(
            self.ep_avg_exclusion) + " | " + "Ep min exclusions: " + str(
            self.ep_min_exclusions) + " | " + "Ep max exclusions: " + str(
            self.ep_max_exclusions) + " | " + "Ep exclusions variance: " + str(self.ep_variance_exclusions))
        print("-------------------------------")
        print("")
        print("")


class SN_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, numb_nodes, connectivity, numb_sources_true, numb_sources_false, max_iterations=30,
                 playing=False,
                 immediate_reward_factor_false_nodes=0.12, immediate_reward_factor_true_nodes=0.008, training_statistics_interval=100):

        super(SN_Env, self).__init__()

        # define gym variables
        self.action_space = spaces.MultiBinary(numb_nodes)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(numb_nodes + 1,), dtype=np.float32)
        self.reward_range = (1, 0)

        # define network variables
        self.numb_nodes = numb_nodes
        self.numb_sourcesTrue = numb_sources_true
        self.numb_sourcesFalse = numb_sources_false
        self.max_iterations = max_iterations
        self.connectivity = connectivity

        # define reward variables
        self.playing = playing
        self.immediate_reward_factor_false_nodes = immediate_reward_factor_false_nodes
        self.immediate_reward_factor_true_nodes = immediate_reward_factor_true_nodes

        # define statistic variables
        self.training_statistics_interval = training_statistics_interval
        self.statistics = Statistics()
        self.ep_training_iterations = 0
        self.exclusion_statistics = np.zeros((self.training_statistics_interval,))

        self._set_up_network()

    def step(self, action):
        # Execute one time step within the environment

        old_state = self.state[self.excluded_members_array == 0]
        action_old_state = action[self.excluded_members_array == 0]
        # Exclude all members where nodeID in action is 1 by setting their state value to zero and deleting all their
        # edges.
        action_inverse = list(map(lambda x: 1 if x == 0 else 0, action))
        action_boolean = list(map(lambda x: True if x == 1 else False, action))

        old_state_excluded = np.multiply(action_inverse, self.state)
        self.state = old_state_excluded
        self.edges[action_boolean, :] = 0
        self.edges[:, action_boolean] = 0

        new_array_excluded = np.minimum(action + self.excluded_members_array, np.ones((self.numb_nodes,), dtype=int))

        self.excluded_members += (np.sum(new_array_excluded) - np.sum(self.excluded_members_array))
        self.excluded_members_array = new_array_excluded

        self.state = self._next_observation()

        if not self.playing:
            reward = self._calculate_reward(action_old_state, old_state)
        else:
            reward = self._calculate_reward_discrete()

        self.iteration += 1

        done = (self.iteration >= self.max_iterations) or (self.excluded_members == self.numb_nodes)

        if not self.playing and done:
            self.exclusion_statistics[self.ep_training_iterations] = self.excluded_members
            self.ep_training_iterations += 1

        return_state = np.zeros((self.numb_nodes + 1,))
        return_state[0:-1] = self.state
        return_state[-1] = self.iteration / self.max_iterations

        return return_state, reward, done, {}

    def reset(self, set_Seed=False, seed=0):
        # Reset the state of the environment to an initial state
        if set_Seed:
            np.random.seed(seed)
        else:
            np.random.seed()
        self._set_up_network()

        # last element of return_state is porgress / current iteration
        return_state = np.zeros((self.numb_nodes + 1,))
        return_state[0:-1] = self.state

        return return_state

    def render(self, mode='human', close=False):
        estimated_value = np.sum(self.state) / (self.numb_nodes - self.excluded_members)
        true_value = self.sources_true_value
        print(f'Estimated value: {estimated_value}')
        print(f'True value: {true_value}')
        print(f'Error: {np.abs(estimated_value - self.sources_true_value)}')

    def _set_up_network(self):

        # update and print statistics
        if not self.playing and self.ep_training_iterations > 0 and (
                self.ep_training_iterations % self.training_statistics_interval) == 0:
            self._update_statistics()

        # Gilbert random graph
        edges_boolean = np.random.random((self.numb_nodes, self.numb_nodes)) < self.connectivity
        self.edges = edges_boolean.astype(int)

        # create external sources
        self.sources_true_value = np.random.uniform(0, 1)
        sources_false_value = self._generate_false_sources(self.sources_true_value, self.numb_sourcesFalse)
        self.sources = np.concatenate((np.repeat(self.sources_true_value, self.numb_sourcesTrue), sources_false_value))

        # create trust in sources
        trust_in_sources_nn = np.zeros((self.numb_nodes, self.sources.shape[0]))

        # trust_in_sources_nn[:, 0:3] = np.random.random((self.numb_nodes, 3)) # trust in 3 sources
        trust_in_sources_nn[:, 0] = np.random.random((self.numb_nodes,))
        self.initial_noise = np.random.normal(0, 0.01, self.numb_nodes)

        [np.random.shuffle(x) for x in trust_in_sources_nn]

        # normalize trust in sources
        self.trust_in_sources = trust_in_sources_nn / trust_in_sources_nn.sum(axis=1)[:, np.newaxis]

        # create trust in network
        self.trust_in_network = np.random.uniform(0, 0.5, self.numb_nodes)

        # create initial state
        self.state = np.zeros((self.numb_nodes,))
        self.state = np.dot(self.trust_in_sources, self.sources)
        self.state += + self.initial_noise  # Add noise to initial state

        self.excluded_members = 0
        self.excluded_members_array = np.zeros((self.numb_nodes,))
        self.initial_noise = np.zeros((self.numb_nodes,))
        self.iteration = 0
        self.false_exclusions = 0

    def _next_observation(self):
        old_state = self.state

        # Get network factor
        network_values = np.dot(self.edges, self.state)
        action_boolean = list(
            map(lambda x: True if x == 1 else False, self.excluded_members_array))  # omit excluded members
        sum_over_edges = self.edges.sum(axis=1)
        # ignore excluded members
        network_values_normalized = np.divide(network_values, sum_over_edges, out=np.zeros_like(network_values),
                                              where=sum_over_edges != 0)
        network_factor = np.multiply(network_values_normalized, self.trust_in_network)

        # Get source factor
        source_values = np.dot(self.trust_in_sources, self.sources) + self.initial_noise
        source_factor = np.multiply(source_values, ([1] * self.numb_nodes - self.trust_in_network))

        # Combine the two factors
        new_observation = network_factor + source_factor

        # in case a member has no neighbours left
        new_observation[sum_over_edges == 0] = old_state[sum_over_edges == 0]
        new_observation[action_boolean] = -1  # set excluded members to value -1

        return new_observation

    def _calculate_reward(self, action, old_state):

        if self.numb_nodes - self.excluded_members > 0:
            # reward over all active nodes
            reward_all_nodes = self._calculate_reward_discrete()

            # extra reward for exclusion of clearly false nodes
            numb_correct_exclusions, numb_missed_exclusions = self._get_missed_exclusions(action, old_state)
            numb_clearly_false_nodes = numb_correct_exclusions + numb_missed_exclusions
            bonus_reward_false_nodes = 2 * numb_correct_exclusions - numb_missed_exclusions
            if numb_clearly_false_nodes > 0:
                bonus_reward_false_nodes_normalized = (bonus_reward_false_nodes / (3 * numb_clearly_false_nodes)) + (
                        1 / 3)
                false_factor = self.immediate_reward_factor_false_nodes * numb_clearly_false_nodes
            else:
                bonus_reward_false_nodes_normalized = 0
                false_factor = 0

            # extra reward for keeping clearly true nodes
            numb_false_exclusions, numb_correct_non_exclusions = self._get_false_exclusions(action, old_state)
            self.false_exclusions += numb_false_exclusions
            numb_clearly_true_nodes = numb_false_exclusions + numb_correct_non_exclusions
            bonus_reward_true_nodes = 2 * numb_correct_non_exclusions - numb_false_exclusions
            if numb_clearly_true_nodes > 0:
                bonus_reward_true_nodes_normalized = (bonus_reward_true_nodes / (3 * numb_clearly_true_nodes)) + (1 / 3)
                true_factor = self.immediate_reward_factor_true_nodes * self.false_exclusions
            else:
                bonus_reward_true_nodes_normalized = 0
                true_factor = 0

            reward = (1 - false_factor - true_factor) * reward_all_nodes + false_factor \
                     * bonus_reward_false_nodes_normalized + true_factor * bonus_reward_true_nodes_normalized

        else:
            reward = 0  # if all members are excluded

        return reward

    def _calculate_reward_discrete(self):
        if self.numb_nodes - self.excluded_members > 0:
            action_boolean = list(map(lambda x: True if x == 0 else False, self.excluded_members_array))
            state_active_members = self.state[action_boolean]
            difference = np.abs(state_active_members - self.sources_true_value)
            difference_discrete = list(map(lambda x: self._get_discrete_reward(x), difference))
            reward_total = np.add.reduce(difference_discrete)
            return reward_total / (self.numb_nodes - self.excluded_members)
        else:
            return 0

    def _get_discrete_reward(self, dif):
        if dif < 0.5:
            return -2 * dif + 1
        else:
            return 0

    def _get_missed_exclusions(self, action, old_state):
        is_false_value = lambda x: True if np.abs((x - self.sources_true_value)) > 0.2 else False
        numb_missed_exclusions = np.sum(
            list(map(lambda x: 1 if is_false_value(x[1]) and x[0] == 0 else 0, zip(action, old_state))))
        numb_correct_exclusions = np.sum(
            list(map(lambda x: 1 if is_false_value(x[1]) and x[0] == 1 else 0, zip(action, old_state))))
        return numb_correct_exclusions, numb_missed_exclusions

    def _get_false_exclusions(self, action, old_state):
        is_true_value = lambda x: True if np.abs((x - self.sources_true_value)) < 0.05 else False
        numb_false_exclusions = np.sum(
            list(map(lambda x: 1 if is_true_value(x[1]) and x[0] == 1 else 0, zip(action, old_state))))
        numb_correct_nonexclusions = np.sum(
            list(map(lambda x: 1 if is_true_value(x[1]) and x[0] == 0 else 0, zip(action, old_state))))
        return numb_false_exclusions, numb_correct_nonexclusions

    # def _calculate_immediate_reward_one_iteration(self, action, state):
    #     sources_imbalance = 1.3 * (self.numb_sourcesTrue / self.numb_sourcesFalse)
    #     is_true_value = lambda x: True if np.abs((x - self.sources_true_value)) < 0.01 else False
    #     correct_action_true = lambda x, y: True if (is_true_value(x) and y == 0) else False
    #     correct_action_false = lambda x, y: True if (not is_true_value(x) and y == 1) else False
    #     rewards = list(map(lambda x: 1 if (correct_action_true(x[0], x[1])) else (
    #         sources_imbalance if (correct_action_false(x[0], x[1]))
    #         else (-sources_imbalance if (not correct_action_false(x[0], x[1]) and not is_true_value(x[0]))
    #               else -1)), zip(state, action)))
    #     best_possible_reward = np.sum(list(map(lambda x: sources_imbalance if not is_true_value(x) else 1, state)))
    #     normalized_reward = (np.sum(rewards) + best_possible_reward) / (2 * best_possible_reward)
    #     return normalized_reward
    #
    # def _calculate_immediate_reward_multiple_iterations(self, action, state):
    #     sources_imbalance = 1.5 * (self.numb_sourcesTrue / self.numb_sourcesFalse)
    #     is_true_value = lambda x: True if np.abs((x - self.sources_true_value)) < 0.01 else False
    #     already_excluded = lambda x: True if x == -1 else False
    #     false_exclusion = lambda x, y: True if (is_true_value(x) and y == 1) else False
    #     false_keeping = lambda x, y: True if (not is_true_value(x) and y == 0) else False
    #     correct_exclusion = lambda x, y: True if (not is_true_value(x) and y == 1) else False
    #     rewards = list(map(lambda x: 0 if already_excluded(x[0]) else (
    #         -1 if false_exclusion(x[0], x[1]) else (-sources_imbalance if false_keeping(x[0], x[1]) else
    #                                                 (sources_imbalance if correct_exclusion(x[0], x[1]) else 1))),
    #                        zip(state, action)))
    #     best_possible_reward = np.sum(
    #         list(map(lambda x: 1 if is_true_value(x) else (0 if (already_excluded(x)) else sources_imbalance), state)))
    #     normalized_reward = (np.sum(rewards) + best_possible_reward) / (2 * best_possible_reward)
    #     return normalized_reward

    def _generate_false_sources(self, true_value, size):
        '''
        Generate false source values with uniform distribution in interval {[0,true_value-0.2],[true_value+0.2,1]}
        :param true_value:
        :param size:
        :return:
        '''
        if true_value - 0.2 > 0:
            lower_interval = np.random.uniform(0, true_value - 0.2, size=size)
            size_li = true_value - 0.2
        else:
            size_li = 0
            lower_interval = np.zeros((size,))
        if true_value + 0.2 < 1:
            upper_interval = np.random.uniform(true_value + 0.2, 1, size=size)
            size_ui = 1 - (true_value + 0.2)
        else:
            size_ui = 0
            upper_interval = np.zeros((size,))

        if size_li == 0:
            prob_li = 0
        elif size_ui == 0:
            prob_li = 1
        else:
            prob_li = size_li / (size_ui + size_li)

        interval_choice = np.random.random(size) < prob_li
        return np.where(interval_choice, lower_interval, upper_interval)

    def _update_statistics(self):
        self.statistics.update_statistics(self.training_statistics_interval, self.exclusion_statistics)
        self.statistics.print_statistics()
        self.ep_training_iterations = 0
        self.exclusion_statistics = np.zeros((self.training_statistics_interval,))
