import torch
import numpy as np
import torch.nn.functional as F
from net.net import VAnet
import copy

class DQN:
    ''' DQN算法 '''
    def __init__(self, vanet, learning_rate, gamma, epsilon, target_update, device):
        self.q_net = vanet
        self.target_q_net = copy.deepcopy(vanet)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.q_net.fc2.out_features)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item() # 转成动作序号
        return action

    def update(self, transition_dict, idx, **kwargs):
        states = transition_dict['states'][:, idx].to(self.device)
        actions = transition_dict['actions'][:, idx].to(torch.int64).to(self.device)
        rewards = transition_dict['rewards'][:, idx].to(self.device)
        next_states = transition_dict['next_states'][:, idx].to(self.device)
        dones = transition_dict['dones'][:, idx].to(torch.int64).to(self.device)
        truncated = transition_dict['truncated'][:, idx].to(torch.int64).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(0))  # 模型预测本状态Q值
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones | truncated)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets.detach()))  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        return dqn_loss.item()
