# Created by Zehong
# three Monte-Carlo algorithms are introduced here
# 1. basic Monte-Carlo(on-policy algorithm)
# 2. Exploring Starts Monte-Carlo
# 3. epsilon-greedy Monte-Carlo

import numpy as np
from examples.arguments import args
import matplotlib.pyplot as plt

class BasicMC:
    def __init__(self, env, gamma=0.9, theta=1e-10, episodes=100, iterations=100):
        self.env = env
        
        self.gamma = gamma
        self.theta = theta
        self.episodes = episodes
        self.iterations = iterations
        
        self.policy = np.zeros((env.num_states, len(env.action_space)))
        self.policy[:, 0] = 1
        self.V = np.zeros(env.num_states)
    
    def iteration(self):
        for s in range(self.env.num_states):
            state = (s % self.env.env_size[0], s // self.env.env_size[0])
            q_values = []
            for a, action in enumerate(self.env.action_space):
                # policy evaluation
                q_value = 0
                state_temp = state
                # sampling the trajectory
                # TODO: what if the behavior policy is stochastic? Is one episode enough to estimate the action value?
                for i in range(self.episodes):
                    next_state, reward = self.env.get_next_state_reward(state_temp, action)
                    q_value += ( self.gamma ** i) * reward
                    state_temp = (next_state %  self.env.env_size[0], next_state //  self.env.env_size[0])
                    # take the action according to the policy
                    # TODO: what if the behavior policy is stochastic?
                    action_idx = np.argmax( self.policy[next_state])
                    action = self.env.action_space[action_idx]
                q_values.append(q_value)
            # policy improvement
            idx = np.argmax(np.array(q_values))
            self.policy[s, idx] = 1
            self.policy[s, np.arange(len(self.env.action_space)) != idx] = 0
            # calculate the state values
            self.V[s] = max(q_values)

# def Basic_MC(canvas, gamma=0.9, theta=1e-10, episodes=100, iterations=1000):
#     '''
#     basic Monte-Carlo algorithm for solving the Bellman Optimality Equation(BOE)
#     :param env: instance of the environment
#     :param gamma: discounted factor
#     :param theta: threshold level for convergence
#     :param episodes: length of episodes
#     :param iterations: number of iterations
#     :return:
#     '''
#     env=canvas.env
#     # initialize the deterministic behavior policy
#     policy = np.zeros((env.num_states, len(env.action_space)))
#     policy[:, 0] = 1
#     V = np.zeros(env.num_states)
#     for k in range(iterations):
#         for s in range(env.num_states):
#             state = (s % env.env_size[0], s // env.env_size[0])
#             q_values = []
#             for a, action in enumerate(env.action_space):
#                 # policy evaluation
#                 q_value = 0
#                 state_temp = state
#                 # sampling the trajectory
#                 # TODO: what if the behavior policy is stochastic? Is one episode enough to estimate the action value?
#                 for i in range(episodes):
#                     next_state, reward = env.get_next_state_reward(state_temp, action)
#                     q_value += (gamma ** i) * reward
#                     state_temp = (next_state % env.env_size[0], next_state // env.env_size[0])
#                     # take the action according to the policy
#                     # TODO: what if the behavior policy is stochastic?
#                     action_idx = np.argmax(policy[next_state])
#                     action = env.action_space[action_idx]
#                 q_values.append(q_value)
#             # policy improvement
#             idx = np.argmax(np.array(q_values))
#             policy[s, idx] = 1
#             policy[s, np.arange(len(env.action_space)) != idx] = 0
#             # calculate the state values
#             V[s] = max(q_values)
#         canvas.update_env(policy, V)
#     return V, policy


class ExploringStartsMC:
    def __init__(self, env, gamma=0.9, theta=1e-10, episodes=1000, iterations=100):
        self.env = env
        
        self.gamma = gamma
        self.theta = theta
        self.episodes = episodes
        self.iterations = iterations
        self.i = 0
        self.policy = np.zeros((env.num_states, len(env.action_space)))
        self.policy[:, np.random.choice(len(env.action_space), env.num_states)] = 1
        # initial guess of the action values
        self.Q = np.zeros((env.num_states, len(env.action_space)))
        self.V = np.zeros(env.num_states)
        self.return_temp = np.zeros((env.num_states, len(env.action_space)))
        self.num_temp = np.zeros((env.num_states, len(env.action_space)))
    
    def iteration(self):
        state_action_pairs = []
        rewards = []
        # TODO: what are the following three lines for?(exploring-starts condition)
        pair_idx = self.i % (self.env.num_states * len(self.env.action_space))
        s = pair_idx // self.env.num_states
        a = pair_idx % len(self.env.action_space)
        state_action_pairs.append((s, a))
        next_state, reward = self.env.get_next_state_reward((s % self.env.env_size[0], s // self.env.env_size[0]), self.env.action_space[a])
        rewards.append(reward)
        for j in range(self.episodes):
            s_temp = next_state
            # sample following the current policy
            action_idx = np.argmax(self.policy[s_temp])
            action = self.env.action_space[action_idx]
            # record the state-action pairs
            state_action_pairs.append((s_temp, action_idx))
            next_state, reward = self.env.get_next_state_reward((s_temp % self.env.env_size[0], s_temp // self.env.env_size[0]),
                                                           action)
            rewards.append(reward)
        # policy evaluation, every-visit method
        # TODO: what if first-visit method is used? (generalized policy iteration)
        g = 0
        for w in range(len(state_action_pairs) - 1, -1, -1):
            g = self.gamma * g + rewards[w]
            s, a = state_action_pairs[w]
            self.return_temp[s, a] += g
            self.num_temp[s, a] += 1
        # policy improvement
        for s in range(self.env.num_states):
            for a, _ in enumerate(self.env.action_space):
                self.Q[s, a] = self.return_temp[s, a] / self.num_temp[s, a] if self.num_temp[s, a] != 0 else 0
            idx = np.argmax(self.Q[s])
            self.policy[s, idx] = 1
            self.policy[s, np.arange(len(self.env.action_space)) != idx] = 0
            # update state values
            self.V[s] = max(self.Q[s])
        self.i += 1



# def ExploringStarts_MC(canvas, gamma=0.9, theta=1e-10, episodes=1000, iterations=1000):
#     '''
#     Exploring Starts Monte-Carlo algorithm for solving the Bellman Optimality Equation(BOE)
#     :param env:
#     :param gamma:
#     :param theta:
#     :param episodes: length of an episode
#     :param iterations:
#     :return:
#     '''
#     env=canvas.env
#     # initialize the deterministic behavior policy
#     policy = np.zeros((env.num_states, len(env.action_space)))
#     policy[:, np.random.choice(len(env.action_space), env.num_states)] = 1
#     # initial guess of the action values
#     Q = np.zeros((env.num_states, len(env.action_space)))
#     V = np.zeros(env.num_states)
#     return_temp = np.zeros((env.num_states, len(env.action_space)))
#     num_temp = np.zeros((env.num_states, len(env.action_space)))
#     for i in range(iterations):
#         # generate an episode and use every-visit method to boost the sampling efficiency
#         state_action_pairs = []
#         rewards = []
#         # TODO: what are the following three lines for?(exploring-starts condition)
#         pair_idx = i % (env.num_states * len(env.action_space))
#         s = pair_idx // env.num_states
#         a = pair_idx % len(env.action_space)
#         state_action_pairs.append((s, a))
#         next_state, reward = env.get_next_state_reward((s % env.env_size[0], s // env.env_size[0]), env.action_space[a])
#         rewards.append(reward)
#         for j in range(episodes):
#             s_temp = next_state
#             # sample following the current policy
#             action_idx = np.argmax(policy[s_temp])
#             action = env.action_space[action_idx]
#             # record the state-action pairs
#             state_action_pairs.append((s_temp, action_idx))
#             next_state, reward = env.get_next_state_reward((s_temp % env.env_size[0], s_temp // env.env_size[0]),
#                                                            action)
#             rewards.append(reward)
#         # policy evaluation, every-visit method
#         # TODO: what if first-visit method is used? (generalized policy iteration)
#         g = 0
#         for w in range(len(state_action_pairs) - 1, -1, -1):
#             g = gamma * g + rewards[w]
#             s, a = state_action_pairs[w]
#             return_temp[s, a] += g
#             num_temp[s, a] += 1
#         # policy improvement
#         for s in range(env.num_states):
#             for a, _ in enumerate(env.action_space):
#                 Q[s, a] = return_temp[s, a] / num_temp[s, a] if num_temp[s, a] != 0 else 0
#             idx = np.argmax(Q[s])
#             policy[s, idx] = 1
#             policy[s, np.arange(len(env.action_space)) != idx] = 0
#             # update state values
#             V[s] = max(Q[s])
#         canvas.update_env(policy, V)
#     return V, policy


class EpsilonGreedyMC:
    def __init__(self, env, epsilon=0.5, gamma=0.9, episodes=1000, iterations=100):
        self.env = env
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.episodes = episodes
        self.iterations = iterations
        self.policy = np.random.rand(env.num_states, len(env.action_space))
        self.policy /= self.policy.sum(axis=1)[:, np.newaxis]
        self.Q = np.zeros((env.num_states, len(env.action_space)))
        self.V = np.zeros(env.num_states)
        self.return_temp = np.zeros((env.num_states, len(env.action_space)))
        self.num_temp = np.zeros((env.num_states, len(env.action_space)))
    
    def iteration(self):
        s = np.random.choice(self.env.num_states)
        a = np.random.choice(len(self.env.action_space))
        pairs = []
        rewards = []
        s_temp = s
        
        for i in range(self.episodes):
            # sample the trajectory following the current policy
            action = self.env.action_space[a]
            pairs.append((s_temp, a))
            next_state, reward = self.env.get_next_state_reward((s_temp % self.env.env_size[0], s_temp // self.env.env_size[0]), action)
            # _, _, _, _ = self.env.step(action)  # observe the trajectories
            rewards.append(reward)
            s_temp = next_state
            # TODO: randomly select an action from the action space, rolete wheel selection
            # a = rolete_wheel_select(self.policy[s_temp].copy())
            a = np.random.choice(len(self.env.action_space), p=self.policy[s_temp])
        # self.env.render()  # play the video
        # policy evaluation, every-visit method
        g = 0
        for t in range(len(rewards) - 1, -1, -1):
            g = self.gamma * g + rewards[t]
            s, a = pairs[t]
            self.return_temp[s, a] += g
            self.num_temp[s, a] += 1
            self.Q[s, a] = self.return_temp[s, a] / self.num_temp[s, a] if self.num_temp[s, a] != 0 else 0
            idx = np.argmax(self.Q[s])
            self.policy[s, idx] = 1 - self.epsilon * (len(self.env.action_space) - 1) / len(self.env.action_space)
            self.policy[s, np.arange(len(self.env.action_space)) != idx] = self.epsilon / len(self.env.action_space)
            self.V[s] = max(self.Q[s])


# def e_greedy_MC(canvas, epsilon=0.5, gamma=0.5, episodes=500, iterations=1000):
#     '''
#     epsilon-greedy Monte-Carlo algorithm for solving the Bellman Optimality Equation(BOE)
#     :param env:
#     :param: epsilon:
#     :param gamma:
#     :param episodes: length of an episode
#     :param iterations:
#     :return:
#     '''
#     env = canvas.env
#     # initialize the stochastic policy
#     policy = np.random.rand(env.num_states, len(env.action_space))
#     policy /= policy.sum(axis=1)[:, np.newaxis]
#     # initial guess of the action values and state values
#     Q = np.zeros((env.num_states, len(env.action_space)))
#     V = np.zeros(env.num_states)
#     # record the frequency of each state-action pair
#     visits = np.zeros((env.num_states, len(env.action_space)))
#     return_temp = np.zeros((env.num_states, len(env.action_space)))
#     num_temp = np.zeros((env.num_states, len(env.action_space)))
#     for k in range(iterations):
#         s = np.random.choice(env.num_states)
#         a = np.random.choice(len(env.action_space))
#         pairs = []
#         rewards = []
#         s_temp = s
#         for i in range(episodes):
#             # sample the trajectory following the current policy
#             action = env.action_space[a]
#             pairs.append((s_temp, a))
#             visits[s_temp, a] += 1
#             next_state, reward = env.get_next_state_reward((s_temp % env.env_size[0], s_temp // env.env_size[0]),
#                                                            action)
#             # _, _, _, _ = env.step(action)  # observe the trajectories
#             rewards.append(reward)
#             s_temp = next_state
#             # TODO: randomly select an action from the action space, rolete wheel selection
#             # a = rolete_wheel_select(policy[s_temp].copy())
#             a = np.random.choice(len(env.action_space), p=policy[s_temp])
#         # env.render()  # play the video
#         # policy evaluation, every-visit method
#         g = 0

#         for t in range(len(rewards) - 1, -1, -1):
#             g = gamma * g + rewards[t]
#             s, a = pairs[t]
#             return_temp[s, a] += g
#             num_temp[s, a] += 1
#             Q[s, a] = return_temp[s, a] / num_temp[s, a] if num_temp[s, a] != 0 else 0
#             idx = np.argmax(Q[s])
#             policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
#             policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
#             V[s] = max(Q[s])

#         canvas.update_env(policy, V)

#     # return the consistent deterministic policy
#     # for s in range(env.num_states):
#     #     idx = np.argmax(policy[s])
#     #     policy[s, idx] = 1
#     #     policy[s, np.arange(len(env.action_space)) != idx] = 0
#     # TODO: calculate the state values of deterministic policy

#     plt.scatter(np.arange(env.num_states * len(env.action_space)), visits.flatten())
#     plt.show()

#     return V, policy
