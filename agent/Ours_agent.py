import torch
import numpy as np
import torch.nn.functional as F

class MacLight:
    def __init__(
        self,
        policy_net,
        critic_net,
        actor_lr: float=1e-4,
        critic_lr: float=5e-3,
        gamma: float=0.9,
        lmbda: float=0.9,
        epochs: int=20,
        eps: float=0.2,
        device: str='cpu',
    ):
        self.muti_agent = False
        self.actor = policy_net.to(device)
        self.critic = critic_net.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # 时序差分学习率
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, agent_name, **kargs):
        states = torch.tensor(transition_dict['states'][agent_name], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'][agent_name], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'][agent_name], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'][agent_name], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'][agent_name], dtype=torch.int).view(-1, 1).to(self.device)
        if len(transition_dict['global_emb']) > 0:
            global_emb = torch.tensor(transition_dict['global_emb'], dtype=torch.float)
        else:
            global_emb = None

        td_target = rewards + self.gamma * self.critic(next_states, global_emb[1:]) * (1 - dones)
        td_delta = td_target - self.critic(states, global_emb[:-1])
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # 所谓的另一个演员就是原来的演员的初始状态
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # 重要性采样系数
            surr1 = ratio * advantage  # 重要性采样
            surr2 = torch.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states, global_emb[:-1]), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    @staticmethod
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float)
        advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-5)
        return advantage_list