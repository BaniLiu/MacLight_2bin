import torch
import numpy as np
import torch.nn.functional as F

class PPO:
    """仅定义网络、优化器和take_action方法"""

    def __init__(
        self,
        policy_net,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        actor_lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.actor = policy_net(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.device = device

    def take_action(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class MAPPO:
    def __init__(
        self,
        agent_name,
        policy_net,
        value_net,
        state_dims: list,
        hidden_dims: list,
        action_dims: list,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        lmbda: float,
        inner_epochs: int,
        eps: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.muti_agent = True
        self.agents = {}
        for i, agt_name in enumerate(agent_name):
            self.agents[agt_name] = PPO(
                policy_net,
                state_dims[i],
                hidden_dims[i],
                action_dims[i],
                actor_lr,
                device=device,
            )
        # self.agents = {'B': PPO, 'E': PPO, ...}
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        self.inner_epochs = inner_epochs
        self.eps = eps
        self.critic = value_net(sum(state_dims), hidden_dims[0]).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update(self, transition_dict: dict, index: int, agent_name: str, **kargs):
        """单智能体更新"""
        states = torch.stack(list(transition_dict['states'].values())).to(self.device)
        actions = torch.tensor(transition_dict['actions'][agent_name], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'][agent_name], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.stack(list(transition_dict['next_states'].values())).to(self.device)
        dones = torch.tensor(transition_dict['dones'][agent_name], dtype=torch.int).view(-1, 1).to(self.device)

        # 转数据格式
        joint_states = states.transpose(1, 0).reshape(states.shape[1], -1)  # 联合状态
        joint_n_states = states.transpose(1, 0).reshape(next_states.shape[1], -1)
        rewards = torch.stack([i.to(torch.float32) for i in rewards])
        actions = torch.stack([i.to(torch.int64).to(self.device) for i in actions])
        dones = torch.stack([i.to(torch.int64) for i in dones])

        td_critic_target = rewards + self.gamma * self.critic(joint_n_states) * (1 - dones)
        td_critic_delta = td_critic_target - self.critic(joint_states)

        td_actor_target = rewards[index] + self.gamma * self.critic(joint_n_states) * (1 - dones[index])
        td_actor_delta = td_actor_target - self.critic(joint_states)

        # 计算优势函数时，用演员自己的 gamma 和 lamba
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_actor_delta.cpu()).to(self.device)

        old_probs = self.agents[agent_name].actor(states[index]).gather(1, actions[index].unsqueeze(0))
        old_log_probs = torch.log(old_probs).detach()
        for _ in range(self.inner_epochs):
            probs = self.agents[agent_name].actor(states[index]).gather(1, actions[index].unsqueeze(0))
            log_probs = torch.log(probs)
            ratio = torch.exp(log_probs - old_log_probs)  # 重要性采样系数
            surr1 = ratio * advantage  # 重要性采样
            surr2 = torch.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            self.agents[agent_name].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.agents[agent_name].actor_optimizer.step()
            critic_loss = torch.mean(F.mse_loss(self.critic(joint_states), td_critic_delta.detach()))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
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