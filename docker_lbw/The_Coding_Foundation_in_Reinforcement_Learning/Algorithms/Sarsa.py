# Created by Zehong Cao

import numpy as np
from examples.arguments import args
import matplotlib.pyplot as plt



class Sarsa:
    def __init__(self, env, start_state=(0, 0), gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
        self.env = env
        self.start_state = start_state
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.iterations = iterations

        self.Q = np.zeros((env.num_states, len(env.action_space)))
        self.V = np.zeros(env.num_states)
        self.policy = np.random.rand(env.num_states, len(env.action_space))
        self.policy /= self.policy.sum(axis=1)[:, np.newaxis]

        self.lengths = []
        self.total_rewards = []

    def iteration(self):
        state = self.start_state  # s
        s = state[1] * self.env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[s])
        action = self.env.action_space[a]  # a
        length = 0
        total_reward = 0
        while state != (4, 4):  # TODO: what if we don't specify the target state?
            # policy evaluation: action value update
            next_state, reward = self.env.get_next_state_reward(state, action)  # r,s
            next_action = np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[next_state])  # a            
            # action value update
            self.Q[s, a] = self.Q[s, a] - self.alpha * (self.Q[s, a] - (reward + self.gamma * self.Q[next_state, next_action]))
            # policy improvement
            idx = np.argmax(self.Q[s])
            # e-greedy exploration
            self.policy[s, idx] = 1 - self.epsilon * (len(self.env.action_space) - 1) / len(self.env.action_space)
            self.policy[s, np.arange(len(self.env.action_space)) != idx] = self.epsilon / len(self.env.action_space)
            # state value update
            self.V[s] = np.sum(self.policy[s] * self.Q[s])
            # update the state and action
            s = next_state
            state = (next_state % self.env.env_size[0], next_state // self.env.env_size[0])
            action = self.env.action_space[next_action]
            a = next_action
            length += 1
            total_reward += reward
        self.lengths.append(length)
        self.total_rewards.append(total_reward)




# def sarsa(canvas, start_state, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
#     '''
#     Sarsa algorithm for solving the Bellman Optimality Equation(BOE)
#     :param env: instance of the environment
#     :param start_state: the start state of the agent
#     :param gamma: discounted factor
#     :param alpha: learning rate
#     :param epsilon: exploration rate
#     :param episodes: length of episodes
#     :param iterations: number of iterations
#     :return:
#     '''
#     env=canvas.env
#     # initialize the state values
#     Q = np.zeros((env.num_states, len(env.action_space)))
#     V = np.zeros(env.num_states)
#     policy = np.random.rand(env.num_states, len(env.action_space))
#     policy /= policy.sum(axis=1)[:, np.newaxis]

#     lengths = []
#     total_rewards = []
#     for k in range(iterations):
#         # TODO: what if the start state is not unique?
#         state = start_state  # s
#         s = state[1] * env.env_size[0] + state[0]
#         a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
#         action = env.action_space[a]  # a

#         length = 0
#         total_reward = 0
#         while state != (4, 4):  # TODO: what if we don't specify the target state?
#             # policy evaluation: action value update
#             next_state, reward = env.get_next_state_reward(state, action)  # r,s
#             next_action = np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])  # a
#             # action value update
#             Q[s, a] = Q[s, a] - alpha * (Q[s, a] - (reward + gamma * Q[next_state, next_action]))
#             # policy improvement
#             idx = np.argmax(Q[s])
#             # e-greedy exploration
#             policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
#             policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
#             # state value update
#             V[s] = np.sum(policy[s] * Q[s])
#             # update the state and action
#             s = next_state
#             state = (next_state % env.env_size[0], next_state // env.env_size[0])
#             action = env.action_space[next_action]
#             a = next_action
#             length += 1
#             total_reward += reward
#         lengths.append(length)
#         total_rewards.append(total_reward)

#         canvas.update_env(policy, V)

    # # TODO: plot the graph of the convergence of the length of episodes and that of the total rewards of episodes
    # fig = plt.subplots(2, 1)
    # plt.subplot(2, 1, 1)
    # plt.plot(lengths)
    # plt.xlabel('Iterations')
    # plt.ylabel('Length of episodes')
    # plt.subplot(2, 1, 2)
    # plt.plot(total_rewards)
    # plt.xlabel('Iterations')
    # plt.ylabel('Total rewards of episodes')
    # plt.show()

    # return V, policy

class ExpectedSarsa:
    def __init__(self, env, start_state=(0, 0), gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
        self.env = env
        self.start_state = start_state
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.iterations = iterations

        self.Q = np.zeros((env.num_states, len(env.action_space)))
        self.V = np.zeros(env.num_states)
        self.policy = np.random.rand(env.num_states, len(env.action_space))
        self.policy /= self.policy.sum(axis=1)[:, np.newaxis]


    def iteration(self):
        state = self.start_state  # s
        s = state[1] * self.env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[s])
        action = self.env.action_space[a]  # a
        while state != (4, 4):  # TODO: what if we don't specify the target state?
            # policy evaluation: action value update
            next_state, reward = self.env.get_next_state_reward(state, action)  # r,s
            # action value update
            self.Q[s, a] = self.Q[s, a] - self.alpha * (self.Q[s, a] - (reward + self.gamma * self.V[next_state]))
            # policy improvement
            idx = np.argmax(self.Q[s])
            # e-greedy exploration
            self.policy[s, idx] = 1 - self.epsilon * (len(self.env.action_space) - 1) / len(self.env.action_space)
            self.policy[s, np.arange(len(self.env.action_space)) != idx] = self.epsilon / len(self.env.action_space)
            # state value update
            self.V[s] = np.sum(self.policy[s] * self.Q[s])
            # update the state and action
            s = next_state
            state = (next_state % self.env.env_size[0], next_state // self.env.env_size[0])
            a = np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[s])
            action = self.env.action_space[a]


# def expected_sarsa(canvas, start_state=(0, 0), gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
#     env=canvas.env
#     Q = np.zeros((env.num_states, len(env.action_space)))
#     V = np.zeros(env.num_states)
#     policy = np.random.rand(env.num_states, len(env.action_space))
#     policy /= policy.sum(axis=1)[:, np.newaxis]
#     for k in range(iterations):
#         # TODO: what if the start state is not unique?
#         state = start_state  # s
#         s = state[1] * env.env_size[0] + state[0]
#         a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
#         action = env.action_space[a]  # a
#         while state != (4, 4):  # TODO: what if we don't specify the target state?
#             # policy evaluation: action value update
#             next_state, reward = env.get_next_state_reward(state, action)  # r,s
#             # action value update
#             Q[s, a] = Q[s, a] - alpha * (Q[s, a] - (reward + gamma * V[next_state]))
#             # policy improvement
#             idx = np.argmax(Q[s])
#             # e-greedy exploration
#             policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
#             policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
#             # state value update
#             V[s] = np.sum(policy[s] * Q[s])
#             # update the state and action
#             s = next_state
#             state = (next_state % env.env_size[0], next_state // env.env_size[0])
#             a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
#             action = env.action_space[a]
#         canvas.update_env(policy, V)
#     return V, policy


class NStepSarsa:
    def __init__(self, env, start_state=(0, 0), steps=3, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
        self.env = env
        self.start_state = start_state
        self.steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.iterations = iterations

        self.Q = np.zeros((env.num_states, len(env.action_space)))
        self.V = np.zeros(env.num_states)
        self.policy = np.random.rand(env.num_states, len(env.action_space))
        self.policy /= self.policy.sum(axis=1)[:, np.newaxis]


    def iteration(self):
        state = self.start_state  # s
        s = state[1] * self.env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[s])
        action = self.env.action_space[a]  # a
        while state != (4, 4):  # TODO: what if we don't specify the target state?
            rewards = 0
            next_state, reward = self.env.get_next_state_reward(state, action)
            rewards = rewards * self.gamma + reward
            next_action = self.env.action_space[np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[next_state])]
            for t in range(self.steps - 2):
                next_state, reward = self.env.get_next_state_reward(
                    (next_state % self.env.env_size[0], next_state // self.env.env_size[0]), next_action)
                rewards = rewards * self.gamma + reward
                next_action = self.env.action_space[np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[next_state])]
            next_s = next_state
            next_a = np.random.choice(np.arange(len(self.env.action_space)), p=self.policy[next_s])
            self.Q[s, a] = self.Q[s, a] - self.alpha * (self.Q[s, a] - (rewards + (self.gamma ** self.steps) * self.Q[next_s, next_a]))
            idx = np.argmax(self.Q[s])
            self.policy[s, idx] = 1 - self.epsilon * (len(self.env.action_space) - 1) / len(self.env.action_space)
            self.policy[s, np.arange(len(self.env.action_space)) != idx] = self.epsilon / len(self.env.action_space)
            self.V[s] = np.sum(self.policy[s] * self.Q[s])
            s = next_s
            state = (next_s % self.env.env_size[0], next_s // self.env.env_size[0])
            a = next_a
            action = self.env.action_space[a]
            
# def n_step_sarsa(canvas, start_state=(0, 0), steps=3, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
#     env=canvas.env
#     Q = np.zeros((env.num_states, len(env.action_space)))
#     V = np.zeros(env.num_states)
#     policy = np.random.rand(env.num_states, len(env.action_space))
#     policy /= policy.sum(axis=1)[:, np.newaxis]
#     for k in range(iterations):
#         state = start_state  # s
#         s = state[1] * env.env_size[0] + state[0]
#         a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
#         action = env.action_space[a]  # a
#         while state != (4, 4):
#             rewards = 0
#             next_state, reward = env.get_next_state_reward(state, action)
#             rewards = rewards * gamma + reward
#             next_action = env.action_space[np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])]
#             for t in range(steps - 2):
#                 next_state, reward = env.get_next_state_reward(
#                     (next_state % env.env_size[0], next_state // env.env_size[0]), next_action)
#                 rewards = rewards * gamma + reward
#                 next_action = env.action_space[np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])]
#             next_s = next_state
#             next_a = np.random.choice(np.arange(len(env.action_space)), p=policy[next_s])
#             Q[s, a] = Q[s, a] - alpha * (Q[s, a] - (rewards + (gamma ** steps) * Q[next_s, next_a]))
#             idx = np.argmax(Q[s])
#             policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
#             policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
#             V[s] = np.sum(policy[s] * Q[s])
#             s = next_s
#             state = (next_s % env.env_size[0], next_s // env.env_size[0])
#             a = next_a
#             action = env.action_space[a]
#         canvas.update_env(policy, V)
#     return V, policy
