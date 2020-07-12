import gym
import numpy as np

from gym import spaces


class SN_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, numb_nodes, connectivity, numb_sources_true, numb_sources_false, max_iterations=30,
                 pre_training=False, pre_training_episodes_1=50000, pre_training_episodes_2=100000, playing=True):
        super(SN_Env, self).__init__()

        self.action_space = spaces.MultiBinary(numb_nodes)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(numb_nodes,), dtype=np.float32)
        self.reward_range = (0, 1)

        # define network
        self.numb_nodes = numb_nodes
        self.numb_sourcesTrue = numb_sources_true
        self.numb_sourcesFalse = numb_sources_false
        self.max_iterations = max_iterations
        self.connectivity = connectivity

        # set up pre-training parameters
        self.pre_training = pre_training
        if pre_training:
            self.pre_training_phase_1 = True
        else:
            self.pre_training_phase_1 = False

        self.pre_training_max_iterations_1 = pre_training_episodes_1
        self.pre_training_max_iterations_2 = pre_training_episodes_2
        self.pre_training_iterations = 0
        self.playing = playing

        self._set_up_network()

    def step(self, action):
        # Execute one time step within the environment

        # Exclude all members where nodeID in action is 1 by setting their state value to zero and deleting all their
        # edges.
        action_inverse = list(map(lambda x: 1 if x == 0 else 0, action))
        action_boolean = list(map(lambda x: True if x == 1 else False, action))
        if self.pre_training_phase_1:
            reward = self._calculate_immediate_reward_multiple_iterations(action, self.state)

        old_state_excluded = np.multiply(action_inverse, self.state)
        self.state = old_state_excluded
        self.edges[action_boolean, :] = 0
        self.edges[:, action_boolean] = 0

        new_array_excluded = np.minimum(action + self.excluded_members_array, np.ones((self.numb_nodes,), dtype=int))
        self.excluded_members += (np.sum(new_array_excluded) - np.sum(self.excluded_members_array))
        self.excluded_members_array = new_array_excluded

        self.state = self._next_observation()

        if not self.pre_training_phase_1:
            reward = self._calculate_reward_discrete()

        self.iteration += 1

        done = (self.iteration >= self.max_iterations) or (self.excluded_members == self.numb_nodes)

        if done and self.pre_training:
            self.pre_training_iterations += 1
            if (self.pre_training_iterations >= self.pre_training_max_iterations_1) and self.pre_training_phase_1:
                self.pre_training_phase_1 = False
            if self.pre_training_phase_1 and self.pre_training_iterations >= self.pre_training_max_iterations_1 + self.pre_training_max_iterations_2:
                self.pre_training = False

        return self.state, reward, done, {}

    def reset(self, set_Seed=False, seed=0):
        # Reset the state of the environment to an initial state
        if set_Seed:
            np.random.seed(seed)
        else:
            np.random.seed()
        self._set_up_network()
        return self.state

    def render(self, mode='human', close=False):
        estimated_value = np.sum(self.state) / (self.numb_nodes - self.excluded_members)
        true_value = self.sources_true_value
        print(f'Estimated value: {estimated_value}')
        print(f'True value: {true_value}')
        print(f'Error: {np.abs(estimated_value - self.sources_true_value)}')

    def _set_up_network(self):
        #Gilbert random graph
        edges_boolean = np.random.random((self.numb_nodes, self.numb_nodes)) < self.connectivity
        self.edges = edges_boolean.astype(int)
        self.sources_true_value = np.random.uniform(0, 1)
        sources_false_value = self._generate_false_sources(self.sources_true_value, self.numb_sourcesFalse)
        self.sources = np.concatenate((np.repeat(self.sources_true_value, self.numb_sourcesTrue), sources_false_value))

        #trust_in_sources_nn = np.random.random((self.numb_nodes, self.sources.shape[0]))

        trust_in_sources_nn = np.zeros((self.numb_nodes, self.sources.shape[0]))
        if self.pre_training and self.pre_training_phase_1:
            trust_in_sources_nn[:, 0] = np.random.random((self.numb_nodes, )) # trust in 1 source
        else:
            trust_in_sources_nn[:, 0:3] = np.random.random((self.numb_nodes, 3)) # trust in 3 sources
        [np.random.shuffle(x) for x in trust_in_sources_nn]

        # normalize trust in sources
        self.trust_in_sources = trust_in_sources_nn / trust_in_sources_nn.sum(axis=1)[:, np.newaxis]
        if self.pre_training:
            self.trust_in_network = np.random.uniform(0, 0, self.numb_nodes) # network factor deactivated
        else:
            self.trust_in_network = np.random.uniform(0, 0.5, self.numb_nodes)

        self.state = np.zeros((self.numb_nodes,))
        self.state = np.dot(self.trust_in_sources, self.sources)

        self.excluded_members = 0
        self.excluded_members_array = np.zeros((self.numb_nodes,))
        self.iteration = 0

    def _next_observation(self):
        network_values = np.dot(self.edges, self.state)

        sum_over_edges = self.edges.sum(axis=1)
        #ignore excluded members
        network_values_normalized = np.divide(network_values, sum_over_edges, out=np.zeros_like(network_values), where=sum_over_edges!=0)

        network_factor = np.multiply(network_values_normalized, self.trust_in_network)
        source_factor = np.multiply(np.dot(self.trust_in_sources, self.sources), ([1] * self.numb_nodes - self.trust_in_network))
        new_observation = network_factor + source_factor
        action_boolean = list(map(lambda x: True if x == 1 else False, self.excluded_members_array)) # omit excluded members
        new_observation[action_boolean] = -1 #set excluded members to value -1
        return new_observation

    def _calculate_reward(self):
        if (self.numb_nodes - self.excluded_members > 0):
            action_boolean = list(map(lambda x: True if x == 0 else False, self.excluded_members_array))
            state_active_members = self.state[action_boolean]
            difference = np.abs(state_active_members - self.sources_true_value)
            reward_elementwise = np.divide([1] * difference.shape[0], difference, out=np.zeros_like(difference), where=difference >= (1/0.01))
            reward_elementwise = list(map(lambda x: x if x > (1/0.01) else 0.01, reward_elementwise))
            reward_elementwise = np.minimum(reward_elementwise, 100)
            reward = np.sum(np.abs(reward_elementwise)) / (self.numb_nodes - self.excluded_members)
        else:
            reward = 0
        return reward

    def _calculate_reward_discrete(self):
        # Helps to stop agent into converging into local minima where no members are excluded at all
        if not self.playing:
            if self.excluded_members == 0:
                return 0
        if (self.numb_nodes - self.excluded_members > 0):
            action_boolean = list(map(lambda x: True if x == 0 else False, self.excluded_members_array))
            state_active_members = self.state[action_boolean]
            difference = np.abs(state_active_members - self.sources_true_value)
            difference_discrete = list(map(lambda x: self._get_discrete_reward(x), difference))
            reward_total = np.add.reduce(difference_discrete)
            return reward_total / (self.numb_nodes - self.excluded_members)
        else:
            reward = -1
        return reward

    def _calculate_immediate_reward_one_iteration(self, action, state):
        sources_imbalance = 1.3 * (self.numb_sourcesTrue / self.numb_sourcesFalse)
        is_true_value = lambda x: True if np.abs((x - self.sources_true_value)) < 0.01 else False
        correct_action_true = lambda x, y: True if (is_true_value(x) and y == 0)  else False
        correct_action_false = lambda x, y: True if (not is_true_value(x) and y == 1) else False
        rewards = list(map(lambda x: 1 if (correct_action_true(x[0], x[1])) else (sources_imbalance if (correct_action_false(x[0], x[1]))
                                              else (-sources_imbalance if (not correct_action_false(x[0], x[1]) and not is_true_value(x[0]))
                                                    else -1)), zip(state, action)))
        best_possible_reward = np.sum(list(map(lambda x: sources_imbalance if not is_true_value(x) else 1, state)))
        normalized_reward = (np.sum(rewards) + best_possible_reward) / (2 * best_possible_reward)
        return normalized_reward

    def _calculate_immediate_reward_multiple_iterations(self, action, state):
        sources_imbalance = 1.5 * (self.numb_sourcesTrue / self.numb_sourcesFalse)
        is_true_value = lambda x: True if np.abs((x - self.sources_true_value)) < 0.01 else False
        already_excluded = lambda x: True if x == -1 else False
        false_exclusion = lambda x, y: True if (is_true_value(x) and y==1) else False
        false_keeping = lambda x, y: True if (not is_true_value(x) and y==0) else False
        correct_exclusion = lambda x, y: True if (not is_true_value(x) and y==1) else False
        rewards = list(map(lambda x: 0 if already_excluded(x[0]) else(-1 if false_exclusion(x[0], x[1]) else (-sources_imbalance if false_keeping(x[0], x[1]) else
                                                (sources_imbalance if correct_exclusion(x[0], x[1]) else 1))), zip(state, action)))
        best_possible_reward = np.sum(list(map(lambda x: 1 if is_true_value(x) else (0 if (already_excluded(x)) else sources_imbalance), state)))
        normalized_reward = (np.sum(rewards) + best_possible_reward) / (2 * best_possible_reward)
        return normalized_reward


    def _get_discrete_reward(self, dif):
        if (dif < 1):
            return -3*dif + 1
        else:
            return -2


    def _generate_false_sources(self, true_value, size):
        '''
        Generate false source values with uniform distribution in interval {[0,true_value-0.2],[true_value+0.2,1]}
        :param true_value:
        :param size:
        :return:
        '''
        if (true_value - 0.2 > 0):
            lower_interval = np.random.uniform(0, true_value - 0.2, size=size)
            size_li = true_value - 0.2
        else:
            size_li = 0
            lower_interval = np.zeros((size,))
        if (true_value + 0.2 < 1):
            upper_interval = np.random.uniform(true_value + 0.2, 1, size=size)
            size_ui = 1 - (true_value + 0.2)
        else:
            size_ui = 0
            upper_interval = np.zeros((size,))

        if (size_li == 0):
            prob_li = 0
        elif (size_ui == 0):
            prob_li = 1
        else:
            prob_li = size_li / (size_ui + size_li)

        interval_choice = np.random.random(size) < prob_li
        return np.where(interval_choice, lower_interval, upper_interval)


