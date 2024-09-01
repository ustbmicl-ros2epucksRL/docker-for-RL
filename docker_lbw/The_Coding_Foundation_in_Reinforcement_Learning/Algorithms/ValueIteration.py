# Created by SupermanCao
# model-based algorithm

# Question: why the state value of the target state is 100 when gamma equals to 0.9

from matplotlib import pyplot as plt
import numpy as np
from examples.arguments import args


class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-10):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.random.uniform(args.reward_forbidden, args.reward_target, env.num_states)
        self.policy = np.zeros((self.env.num_states, len(self.env.action_space)))
        self.iter_count = 0
        self.delta = 0

    def iteration(self):
        self.iter_count += 1
        self.delta = 0
        # policy update
        for s in range(self.env.num_states):
            state = (s % self.env.env_size[0], s // self.env.env_size[0])
            v = self.V[s]
            q_values = []
            for a, action in enumerate(self.env.action_space):
                next_state, reward = self.env.get_next_state_reward(state, action)
                # TODO: calculate the action values
                q_values.append(reward + self.gamma * self.V[next_state])  # v_k
            # TODO: finish the policy update
            max_idx = np.argmax(q_values)
            self.policy[s, max_idx] = 1
            self.policy[s, np.arange(len(self.env.action_space)) != max_idx] = 0
            # value update
            # TODO: finish the value update
            self.V[s] = max(q_values)  # v_{k+1}
            self.delta = max(self.delta, abs(v - self.V[s]))  # compare the largest gap which corresponds to the infinite norm

        # self.canvas.update_env(self.policy, self.V)
        print(f"Iteration {self.iter_count}, delta: {self.delta}")
        return self.iter_count, self.delta





# def value_iteration(canvas,gamma=0.9, theta=1e-10):
#     '''
#     value iteration for solving the Bellman Optimality Equation(BOE)
#     :param env: instance of the environment
#     :param gamma: discounted factor
#     :param theta: threshold level for convergence
#     :return:
#     '''
#     env = canvas.env
#     # initialize the state values
#     V = np.random.uniform(args.reward_forbidden, args.reward_target, env.num_states)
#     policy = np.zeros((env.num_states, len(env.action_space)))
#     iter_count = 0
#     while True:
#         iter_count += 1
#         delta = 0
#         # policy update
#         for s in range(env.num_states):
#             state = (s % env.env_size[0], s // env.env_size[0])
#             v = V[s]
#             q_values = []
#             for a, action in enumerate(env.action_space):
#                 next_state, reward = env.get_next_state_reward(state, action)
#                 # TODO: calculate the action values
#                 q_values.append(reward + gamma * V[next_state])  # v_k
#             # TODO: finish the policy update
#             max_idx = np.argmax(q_values)
#             policy[s, max_idx] = 1
#             policy[s, np.arange(len(env.action_space)) != max_idx] = 0
#             # value update
#             # TODO: finish the value update
#             V[s] = max(q_values)  # v_{k+1}
#             delta = max(delta, abs(v - V[s]))  # compare the largest gap which corresponds to the infinite norm

#         canvas.update_env(policy, V)
#         print(f"Iteration {iter_count}, delta: {delta}")

#         if delta < theta:
#             break
#     return V, policy
