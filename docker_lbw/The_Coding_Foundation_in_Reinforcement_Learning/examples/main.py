# Created by Zehong Cao

import os
import sys
import time
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
sys.path.append("./")
from src.grid_world import GridWorld
import gym
import random
import numpy as np
from Algorithms.ValueIteration import ValueIteration
from Algorithms.PolicyIteration import PolicyIteration
from Algorithms.Monte_Carlo import BasicMC, ExploringStartsMC, EpsilonGreedyMC
from Algorithms.Sarsa import Sarsa, ExpectedSarsa, NStepSarsa
from Algorithms.Q_Learning import *
from Algorithms.PolicyGradient import *


class ui:
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Reinforcement Learning")
        self.window.geometry("760x400")
        self.graid_size = 5
        self.train_method = 0
        self.theta = 1e-10
        self.delta = 1
        self.method = None
        self.update_interval = 1000
        self.iteration_steps = 0
        # 需要加上这句，不然关闭窗口时程序不退出
        self.window.protocol("WM_DELETE_WINDOW", self._quit)
        self.after_id = None

    def _quit(self):
        self.window.quit()
        self.window.destroy() 


    def method_selection_frame(self):

        frame = tk.Frame(self.window, width=360, height=120)
        frame.place(x=0, y=0, width=360, height=120)

        label = tk.Label(frame, text="选择方法：")
        label.grid(row=0, column=0, sticky="w")

        method_list = ["Value Iteration", "Policy Iteration", "Basic MC", "ExploringStarts MC", "E-Greedy MC", "Sarsa", "Expected Sarsa", "N-Step Sarsa", "On-Policy QL"]

        self.train_method = tk.IntVar()

        for i in range(len(method_list)):
            rb = tk.Radiobutton(frame, text=method_list[i], variable=self.train_method, value=i+1)
            rb.grid(row=int(i / 3) + 1, column=i % 3, sticky="w")


    def start_train(self):

        method_list = [ValueIteration, PolicyIteration, BasicMC, ExploringStartsMC, EpsilonGreedyMC, Sarsa, ExpectedSarsa, NStepSarsa, on_policy_ql]
        if self.train_method.get() >= 5 and self.train_method.get() <= 8:
            self.method = method_list[self.train_method.get() - 1](self.env)
        elif self.train_method.get() > 0 and self.train_method.get() <= 4:
            self.method = method_list[self.train_method.get() - 1](self.env, theta=self.theta)
        elif self.train_method.get() == 0:
            messagebox.showerror("error", "请选择训练方法")
        else:
            self.method = method_list[self.train_method.get() - 1]
        self.update_env()


    def param_frame(self):
        frame = tk.Frame(self.window,width=360, height=120)
        frame.place(x=0, y=120, width=360, height=120)

        label = tk.Label(frame, text="VI、PI收敛阈值(defalut: 1e-10)：")
        label.grid(row=0, column=0, sticky="w")
        # 创建一个文本框，用户可以在里面输入文本
        self.theta_entry = tk.Entry(frame, width=5)
        self.theta_entry.grid(row=0, column=1, padx=5)
        # 创建一个按钮，用户点击时会触发on_submit函数
        submit_button = tk.Button(frame, text="确定", command=self.theta_submit)
        submit_button.grid(row=0, column=2)

        label = tk.Label(frame, text="迭代次数(default: 100)：")
        label.grid(row=1, column=0, sticky="w")
        # 创建一个文本框，用户可以在里面输入文本
        self.theta_entry = tk.Entry(frame, width=5)
        self.theta_entry.grid(row=1, column=1, padx=5)
        # 创建一个按钮，用户点击时会触发on_submit函数
        submit_button = tk.Button(frame, text="确定", command=self.mc_iterations_submit)
        submit_button.grid(row=1, column=2)

        label = tk.Label(frame, text="更新频率(ms,defalut: 1000)：")
        label.grid(row=2, column=0, sticky="w")
        # 创建一个文本框，用户可以在里面输入文本
        self.interval_entry = tk.Entry(frame, width=5)
        self.interval_entry.grid(row=2, column=1, padx=5)
        # 创建一个按钮，用户点击时会触发on_submit函数
        submit_button = tk.Button(frame, text="确定", command=self.interval_submit)
        submit_button.grid(row=2, column=2)

        # 创建文本框
        self.text_box = tk.Text(frame, width=20, height=10)  # 宽度为 40，高度为 10
        self.text_box.grid(row=3, column=0, sticky="nw")  # 添加到主窗口并自动调整大小
        self.update_steps()

        # 创建一个按钮，用户点击时会触发on_submit函数
        reset_button = tk.Button(frame, text="重置数据", command=self.reset_env)
        reset_button.grid(row=3, column=1, sticky="nw")
        # 创建一个按钮，用户点击时会触发on_submit函数
        submit_button = tk.Button(frame, text="开始训练", command=self.start_train)
        submit_button.grid(row=3, column=2, sticky="nw")


    def grid_world_frame(self):
        frame = tk.Frame(self.window,width=400, height=400)
        frame.place(x=360, y=0, width=400, height=400)
        # 构建迷宫环境
        self.env = GridWorld()
        self.env.reset()
        self.env.render()
        # 创建一个 FigureCanvasTkAgg 实例，将 Matplotlib 图形嵌入到 Tkinter
        self.canvas = FigureCanvasTkAgg(self.env.canvas, master=frame)  # `master` 是 Tkinter 窗口或 Frame
        # env.render()
        # plt.pause(1)  # 为了让画布更新
        # 将画布的 widget 添加到 Tkinter 布局中
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack()

    def theta_submit(self):
        # 获取输入值并转换为整数
        try:
            theta = float(self.theta_entry.get())
        except ValueError:
            messagebox.showerror("error", "请输入一个有效的数")
            return

        if isinstance(self.method, ValueIteration) or isinstance(self.method, PolicyIteration):
            # 检查输入值是否在0到1000之间
            if 0 < theta < 1:
                self.theta = theta
            else:
                messagebox.showwarning("warning", "请填写范围在(0, 1)之间的数值的数值")

    def interval_submit(self):
        # 获取输入值并转换为整数
        try:
            update_interval = int(self.interval_entry.get())
        except ValueError:
            messagebox.showerror("error", "请输入一个有效的数")
            return

        # 检查输入值是否在0到1000之间
        if 10 <= update_interval <= 5000:
            self.update_interval = update_interval
        else:
            messagebox.showwarning("warning", "请填写范围在[10, 5000]之间的数值的数值")

    def mc_iterations_submit(self):
        # 获取输入值并转换为整数
        try:
            iterations = int(self.theta_entry.get())
        except ValueError:
            messagebox.showerror("error", "请输入一个有效的数")
            return

        if isinstance(self.method, BasicMC) or isinstance(self.method, ExploringStartsMC) or isinstance(self.method, EpsilonGreedyMC):
        # 检查输入值是否在0到1000之间
            if 1 <= iterations <= 10000:
                self.method.iterations = iterations
            else:    
                messagebox.showwarning("warning", "请填写范围在[1, 10000]之间的数值的数值")

    def update_env(self):
        if self.train_method.get() < 9:
            if self.train_method.get() <= 2:
                if self.delta >= self.theta:
                    self.iteration_steps, self.delta = self.method.iteration()
            elif self.train_method.get() >= 3 and self.train_method.get() <= 8:
                if self.iteration_steps < self.method.iterations:
                    self.iteration_steps += 1
                    self.method.iteration()
            self.env.add_policy(self.method.policy)
            self.env.add_state_values(self.method.V)
            self.env.render()
            self.canvas.draw()
            self.env.reset_policy_and_values()
            self.update_steps()
            self.after_id = self.window.after(self.update_interval, self.update_env)

        else:
            optimal_value, optimal_policy = self.method(self.env)
            self.env.add_policy(optimal_policy)
            self.env.add_state_values(optimal_value)
            self.env.render()
            self.canvas.draw()
            self.update_steps()


    def update_steps(self):
        self.text_box.config(state=tk.NORMAL)
        # 清除文本框中的现有文本
        self.text_box.delete(1.0, tk.END)
        # 插入新的文本
        self.text_box.insert("end",  "迭代次数：" + str(self.iteration_steps))
        self.text_box.config(state=tk.DISABLED)

    def reset_env(self):
        self.grid_world_frame()
        self.iteration_steps = 0
        self.delta = 1
        self.window.after_cancel(self.after_id)
        self.method = None
        self.update_steps()





if __name__ == "__main__":
    user_interface = ui()
    # user_interface.grid_size_frame()
    user_interface.method_selection_frame()
    user_interface.param_frame()
    user_interface.grid_world_frame()
    user_interface.window.mainloop()
    # TODO: Step 1, Choose your environment
    #  (the dpg() function is only compatible with the Pendulum environment since there is a continuous action space)
    #  (the a2c() function is only compatible with the CartPole environment since there is a state space with more observations)
    # 1.1. discrete action space
    # env = GridWorld()
    # env = gym.make('CartPole-v1', render_mode='human')

    # 1.2. continuous action space
    # env = gym.make('Pendulum-v1', render_mode='human')

    # WARNING: Do not modify the following single line

    # TODO: Step 2, Choose your algorithm to test all the algorithms mentioned in the book are implemented as follows

    # check the differences between value iteration and policy iteration
    # optimal_value, optimal_policy = value_iteration(canvas=user_interface)  # the ground truth state values
    # optimal_value, optimal_policy = policy_iteration(env)

    # check the differences among variants of basic Monte-Carlo Methods
    # optimal_value,optimal_policy = Basic_MC(env)
    # optimal_value, optimal_policy = ExploringStarts_MC(env)
    # optimal_value, optimal_policy = e_greedy_MC(env)

    # optimal_value, optimal_policy = sarsa(env, (0, 0))
    # optimal_value, optimal_policy = expected_sarsa(env, start_state=(0, 0))
    # optimal_value, optimal_policy = n_step_sarsa(env, start_state=(0, 0), steps=3)

    # optimal_value, optimal_policy = on_policy_ql(env, start_state=(0, 0))
    # optimal_value, optimal_policy = dqn(env, gt_state_values=optimal_value) # you could obtain the ground_truth state values from value_iteration() above
    # optimal_value, optimal_policy = reinforce(env)
    # optimal_value, optimal_policy = qac(env)
    # optimal_value, optimal_policy = a2c(env)
    # optimal_value, optimal_policy = off_policy_ac(env)
    # optimal_value, optimal_policy = dpg(env)

    # TODO: Step 3, Visualize the eventual policy by rendering the environment
    # if isinstance(user_interface.env, grid_world.GridWorld):
    #     # user_interface.env.render()
    #     # user_interface.env.add_policy(optimal_policy)
    #     # user_interface.env.add_state_values(optimal_value)

    #     user_interface.env.render(animation_interval=5)
