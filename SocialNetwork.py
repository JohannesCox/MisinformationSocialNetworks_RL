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
        print("\n\n-------------------------------")
        print("Exclusion Statistics:")
        print("Total iterations " + str(self.iteration) + " | " + "Moving avg exclusions: " + str(
            self.moving_avg_exclusions) + " | " + "Ep exclusions avg: " + str(
            self.ep_avg_exclusion) + " | " + "Ep min exclusions: " + str(
            self.ep_min_exclusions) + " | " + "Ep max exclusions: " + str(
            self.ep_max_exclusions) + " | " + "Ep exclusions variance: " + str(self.ep_variance_exclusions))
        print("-------------------------------\n\n")


class SN_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, numb_nodes, connectivity, numb_sources_true, numb_sources_false, max_iterations=30,
                 excluding_decision_boundary=0.1, initial_noise_variance=0.05, playing=False, display_statistics=True,
                 training_statistics_interval=100):

        super(SN_Env, self).__init__()

        # define gym variables
        self.action_space = spaces.Box(low=0, high=1, shape=(numb_nodes,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(numb_nodes + 1,), dtype=np.float32)
        self.reward_range = (-1, 1)

        # define network variables
        self.numb_nodes = numb_nodes
        self.numb_sourcesTrue = numb_sources_true
        self.numb_sourcesFalse = numb_sources_false
        self.max_iterations = max_iterations
        self.connectivity = connectivity
        self.excluded_members_array = np.zeros((numb_nodes,))
        self.state = np.zeros((numb_nodes,))
        self.initial_noise_variance = initial_noise_variance

        # define reward variables
        self.playing = playing
        self.excluding_decision_boundary = excluding_decision_boundary

        # define statistic variables
        self.training_statistics_interval = training_statistics_interval
        self.statistics = Statistics()
        self.ep_training_iterations = 0
        self.exclusion_statistics = np.zeros((self.training_statistics_interval,))
        self.display_statistics = display_statistics

        self._set_up_network()

    def step(self, action_box):
        # Execute one time step within the environment

        def hard_desc(x): return 1 if x < self.excluding_decision_boundary else 0

        vec_hard_desc = np.vectorize(hard_desc)
        action_hard_decision = vec_hard_desc(action_box)
        action_hard_decision = action_hard_decision.astype(np.int32)

        # Exclude all members where nodeID in action is 1 by setting their state value to zero and deleting all their
        # edges.
        def action_inverse_func(x): return 1 if x == 0 else 0

        vec_inverse_func = np.vectorize(action_inverse_func)
        action_inverse = vec_inverse_func(action_hard_decision)

        def act_bool(x): return True if x == 1 else False

        vec_act_bool = np.vectorize(act_bool)
        action_boolean = vec_act_bool(action_hard_decision)

        old_state_excluded = np.multiply(action_inverse, self.state)
        self.state = old_state_excluded
        self.edges[action_boolean, :] = 0
        self.edges[:, action_boolean] = 0

        new_array_excluded = np.minimum(action_hard_decision + self.excluded_members_array,
                                        np.ones((self.numb_nodes,), dtype=int))

        self.excluded_members += (np.sum(new_array_excluded) - np.sum(self.excluded_members_array))
        self.excluded_members_array = new_array_excluded

        self.state = self._next_observation()

        reward = self._calculate_reward(action_box)

        self.iteration += 1

        done = (self.iteration >= self.max_iterations) or (self.excluded_members == self.numb_nodes)

        if self.display_statistics and done:
            self.exclusion_statistics[self.ep_training_iterations] = self.excluded_members
            self.ep_training_iterations += 1

        return_state = np.zeros((self.numb_nodes + 1,))
        return_state[0:-1] = self.state
        return_state[-1] = self.iteration / self.max_iterations

        return return_state, reward, done, {}

    def reset(self, set_seed=False, seed=0):
        # Reset the state of the environment to an initial state
        if set_seed:
            np.random.seed(seed)
        else:
            np.random.seed()
        self._set_up_network()

        # last element of return_state is progress / current iteration
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
        if self.display_statistics and self.ep_training_iterations > 0 and (
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

        trust_in_sources_nn[:, 0] = np.random.random((self.numb_nodes,))  # trust in 1 sources
        self.initial_noise = np.random.normal(0, self.initial_noise_variance, self.numb_nodes)

        [np.random.shuffle(x) for x in trust_in_sources_nn]

        # normalize trust in sources
        self.trust_in_sources = trust_in_sources_nn / trust_in_sources_nn.sum(axis=1)[:, np.newaxis]

        # create trust in network
        self.trust_in_network = np.random.uniform(0, 0.5, self.numb_nodes)

        # create initial state
        self.state = np.zeros((self.numb_nodes,))
        self.state = np.dot(self.trust_in_sources, self.sources)
        self.state += + self.initial_noise  # Add noise to initial state
        self.state = np.clip(self.state, 0, 1)

        self.excluded_members = 0
        self.excluded_members_array = np.zeros((self.numb_nodes,), dtype=np.int32)
        self.initial_noise = np.zeros((self.numb_nodes,))
        self.iteration = 0
        self.false_exclusions = 0

    def _next_observation(self):
        old_state = self.state

        # Get network factor
        network_values = np.dot(self.edges, self.state)

        def act_bool(x): return True if x == 1 else False

        vec_act_bool = np.vectorize(act_bool)
        action_boolean = vec_act_bool(self.excluded_members_array)  # omit excluded members
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

    def _generate_false_sources(self, true_value, size):
        if true_value - 0.4 > 0:
            lower_interval = np.random.uniform(0, true_value - 0.4, size=size)
            size_li = true_value - 0.4
        else:
            size_li = 0
            lower_interval = np.zeros((size,))
        if true_value + 0.4 < 1:
            upper_interval = np.random.uniform(true_value + 0.4, 1, size=size)
            size_ui = 1 - (true_value + 0.4)
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

    def _calculate_reward(self, action_box):
        if self.excluded_members < self.numb_nodes:
            trust = action_box[self.excluded_members_array == 0]
            values = self.state[self.excluded_members_array == 0]
            estimated_value = np.sum(np.multiply(trust, values))
            estimated_value_normalized = estimated_value / np.sum(trust)
            reward = self._get_linear_reward(np.abs(estimated_value_normalized - self.sources_true_value))
            return reward
        else:
            return -1

    def _get_linear_reward(self, dif):
        if dif < 0.5:
            return -2 * dif + 1
        else:
            return 0
