"""
CartPole PID 和 DQN 控制示例
姓名：张三
学号：20230001
微信：zhangsan123
"""
import gym
import numpy as np
from stable_baselines3 import DQN  # 需安装stable_baselines3

def pid_control(state, params):
    """PID控制器：返回离散动作0/1"""
    angle, vel = state[2], state[3]
    force = params['Kp']*angle + params['Ki']*params['integral'] + params['Kd']*vel
    params['integral'] += angle
    return 1 if np.clip(force, -10, 10) > 0 else 0

# 初始化环境和参数
env = gym.make('CartPole-v1')
pid_params = {'Kp': 1.5, 'Ki': 0.01, 'Kd': 0.1, 'integral': 0}

# === 方法1：PID控制 ===
state = env.reset()
for _ in range(200):  # PID测试200步
    env.render()
    action = pid_control(state, pid_params)
    state, _, done, _ = env.step(action)
    if done: break

# === 方法2：DQN训练与测试 ===
model = DQN('MlpPolicy', env, verbose=0).learn(5000)  # 快速训练
state = env.reset()
for _ in range(200):  # DQN测试200步
    env.render()
    action, _ = model.predict(state, deterministic=True)
    state, _, done, _ = env.step(action)
    if done: break

env.close()
        
