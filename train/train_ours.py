import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_ours_agent(
    env: object,
    agents: object,
    agent_name: list,
    vae: object,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
):
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor_loss_list = []
    critic_loss_list = []
    vae_loss_list = []
    return_list = []
    waiting_list = []
    queue_list = []
    speed_list = []
    time_list = []
    seed_list = []
    start_time = time.time()
    best_score = -1e10
    actor_best_weight = {}
    critic_best_weight = {}
    optimizer = optim.Adam(vae.parameters(), lr=1e-3) if vae else None

    for episode in range(total_episodes):
        epi_training = False
        transition_dict = {"states": {agt_name: 0 for agt_name in agent_name},
                           "actions": {agt_name: 0 for agt_name in agent_name},
                           "next_states": {agt_name: 0 for agt_name in agent_name},
                           "rewards": {agt_name: 0 for agt_name in agent_name},
                           "dones": {agt_name: 0 for agt_name in agent_name},
                           "global_emb": []}
        episode_return = 0

        # * ---- execute simulation ----
        state, done, truncated = env.reset(seed=seed)[0], False, False
        while not done | truncated:
            action = {}
            for agt_name in agent_name:
                action[agt_name] = agents[agt_name].take_action(state[agt_name])
            next_state, reward, done, truncated, info = env.step(action)
            transition_dict = update_transition(agent_name, epi_training, transition_dict, state,
                                                    done, action, next_state, reward)
            epi_training = True
            state = next_state
            episode_return += np.mean(list(reward.values()))
            done = all(list(done.values()))
            truncated = all(list(truncated.values()))

        # * ---- log to dict  ----
        return_list.append(episode_return)
        waiting_list.append(info[agent_name[0]]["system_total_waiting_time"])
        queue_list.append(info[agent_name[0]]["system_total_stopped"])
        speed_list.append(info[agent_name[0]]["system_mean_speed"])
        time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
        seed_list.append(seed)

        # *  ---- train_vae ----
        if vae:
            whole_state = torch.stack(list(transition_dict['states'].values())).transpose(1, 0)  # 100*16*33
            end_state = torch.stack(list(transition_dict['next_states'].values())).transpose(1, 0)[-1]
            whole_state = torch.cat([whole_state, end_state.unsqueeze(0)])
            reshaped_state = reshape_whole_state(whole_state)
            vae_loss = train_vae(vae, optimizer, reshaped_state)
            dataset = TensorDataset(reshaped_state)
            dataloader = DataLoader(dataset, 128, shuffle=False)
            global_emb = []
            for data in dataloader:
                global_emb.append(vae.representation(data[0].to(device)))
            global_emb_tensor = torch.cat(global_emb)
            transition_dict['global_emb'] = global_emb_tensor
        else:
            vae_loss = None
        vae_loss_list.append(vae_loss)
        # * ---- update agent ----
        for agt_name in agent_name:  # 更新网络
            actor_loss, critic_loss = agents[agt_name].update(transition_dict, agt_name)
            actor_loss_list.append(actor_loss)  # 所有agent的loss放一起了
            critic_loss_list.append(critic_loss)

        # read best weights
        if episode_return > best_score:
            for agt_name in agent_name:
                actor_best_weight[agt_name] = agents[agt_name].actor.state_dict()
                critic_best_weight[agt_name] = agents[agt_name].critic.state_dict()
            best_score = episode_return

        # read best weight
        if epi_training:
            for agt_name in agent_name:
                agents[agt_name].actor.load_state_dict(actor_best_weight[agt_name])
                agents[agt_name].critic.load_state_dict(critic_best_weight[agt_name])

        # save log to file and report train status
        evaluator.evaluate_and_save(writer, return_list, waiting_list, queue_list, speed_list,
                                    time_list, seed_list, ckpt_path, episode, agents, seed,
                                    actor_loss_list, critic_loss_list, vae_loss_list=vae_loss_list, vae=vae)

    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {(total_time / 60):.2f} min")

    return return_list, total_time // 60

def update_transition(agent_name, epi_training, transition_dict, state, done, action, next_state, reward):
    '''
    Since the state and other data of each agent are stored separately in the form of dictionaries, they are merged here.
    '''
    for key, element in zip(['states', 'actions', 'next_states', 'rewards', 'dones'],
                            [state, action, next_state, reward, done]):
        for agt_name in agent_name:
            if not epi_training:
                transition_dict[key][agt_name] = torch.tensor(element[agt_name]).unsqueeze(0)
            else:
                transition_dict[key][agt_name] = torch.cat([transition_dict[key][agt_name],
                                                            torch.tensor([element[agt_name]])])
    return transition_dict

def reshape_whole_state(state):
    global_state = torch.cat([torch.stack([state[:, 3], state[:, 7], state[:, 11], state[:, 15]], dim=1).unsqueeze(1),
                              torch.stack([state[:, 2], state[:, 6], state[:, 10], state[:, 14]], dim=1).unsqueeze(1),
                              torch.stack([state[:, 1], state[:, 5], state[:, 9],  state[:, 13]], dim=1).unsqueeze(1),
                              torch.stack([state[:, 0], state[:, 4], state[:, 8],  state[:, 12]], dim=1).unsqueeze(1)], dim=1)
    global_state = global_state.permute(0, 3, 1, 2)
    return global_state

# VAE 损失函数
def loss_function(recon_x, x, mu, logvar):
    # 重建损失 (MSE 或 BCE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train_vae(model, optimizer, data, epochs=30):
    model.train()
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    loss_list = []
    for _ in range(epochs):
        total_loss = 0
        for i_data in dataloader:
            i_data = i_data[0].to(device)  # 将数据加载到 GPU 或 CPU
            optimizer.zero_grad()
            # 前向传播
            recon_batch, mu, logvar = model(i_data)
            # 计算损失
            loss = loss_function(recon_batch, i_data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        loss_list.append(total_loss)
    return sum(loss_list) / len(loss_list)