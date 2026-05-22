import time
import numpy as np
import torch
from tqdm import tqdm, trange


def train_IPPO_agent(
    env: object,
    agents: object,
    agent_name: list,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
):
    actor_loss_list = []
    critic_loss_list = []
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
    for episode in range(total_episodes):
        epi_training = False
        transition_dict = {"states": {agt_name: 0 for agt_name in agent_name},
                           "actions": {agt_name: 0 for agt_name in agent_name},
                           "next_states": {agt_name: 0 for agt_name in agent_name},
                           "rewards": {agt_name: 0 for agt_name in agent_name},
                           "dones": {agt_name: 0 for agt_name in agent_name}}
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

        # load best weight
        if epi_training:
            for agt_name in agent_name:
                agents[agt_name].actor.load_state_dict(actor_best_weight[agt_name])
                agents[agt_name].critic.load_state_dict(critic_best_weight[agt_name])

        # save log to file and report train status
        evaluator.evaluate_and_save(writer, return_list, waiting_list, queue_list, speed_list,
                                    time_list, seed_list, ckpt_path, episode, agents, seed,
                                    actor_loss_list, critic_loss_list)

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

def train_MAPPO_agent(
    env: object,
    agent: object,
    agent_name: list,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
):
    actor_loss_list = []
    critic_loss_list = []
    return_list = []
    waiting_list = []
    queue_list = []
    speed_list = []
    time_list = []
    seed_list = []
    start_time = time.time()
    best_score = -1e10

    # * ---- execute simulation ----
    for episode in range(total_episodes):
        epi_training = False
        transition_dict = {"states": {agt_name: 0 for agt_name in agent_name},
                           "actions": {agt_name: 0 for agt_name in agent_name},
                           "next_states": {agt_name: 0 for agt_name in agent_name},
                           "rewards": {agt_name: 0 for agt_name in agent_name},
                           "dones": {agt_name: 0 for agt_name in agent_name}}
        episode_return = 0

        state, done, truncated = env.reset()[0], False, False
        while not (done | truncated):
            action = {}
            for agt_name in agent_name:
                action[agt_name] = agent.agents[agt_name].take_action(state[agt_name])
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

        # * ---- update agent ----
        for index, agt_name in enumerate(agent_name):
            actor_loss, critic_loss = agent.update(transition_dict, index, agt_name)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)

        # 保存最佳权重
        if episode_return > best_score:
            actor_best_weight = {
                agt_name: agent.agents[agt_name].actor.state_dict()
                for agt_name in agent.agents
            }
            critic_best_weight = agent.critic.state_dict()
            best_score = episode_return

        if epi_training:
            for agt_name in agent_name:
                agent.agents[agt_name].actor.load_state_dict(actor_best_weight[agt_name])
                agent.critic.load_state_dict(critic_best_weight)

        evaluator.evaluate_and_save(writer, return_list, waiting_list, queue_list, speed_list,
                                    time_list, seed_list, ckpt_path, episode, agent, seed,
                                    actor_loss_list, critic_loss_list)

    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {(total_time / 60):.2f} min")

    return return_list, total_time // 60


def train_IDQN_agent(
    env: object,
    agents: object,
    agent_name: list,
    replay_buffer: object,
    minimal_size: int,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
    **kwargs,
):
    device = kwargs.get('device', 'cpu')
    Qnet_loss_list = []
    return_list = []
    waiting_list = []
    queue_list = []
    speed_list = []
    time_list = []
    seed_list = []
    start_time = time.time()
    best_score = -1e10
    best_weight = {}
    for episode in range(total_episodes):
        epi_training = False
        episode_return = 0

        # * ---- execute simulation ----
        state, done, truncated = env.reset(seed=seed)[0], False, False
        while not (done | truncated):
            action = {}
            for agt_name in agent_name:
                action[agt_name] = agents[agt_name].take_action(state[agt_name])
            next_state, reward, done, truncated, info = env.step(action)
            # 1. 存经验
            next_state_store = torch.tensor(list(next_state.values()))
            state_store = torch.tensor(list(state.values()))
            action_store = torch.tensor(list(action.values()))
            reward_store = torch.tensor(list(reward.values()))
            done_store = torch.tensor(list(done.values()))
            truncated_store = torch.tensor(list(truncated.values()))
            replay_buffer.store(state_store, action_store, reward_store,
                                next_state_store, done_store, truncated_store)
            # 2.更新权重
            if replay_buffer.size > minimal_size:
                for idx, agt_name in enumerate(agent_name):
                    train_data = replay_buffer.sample_batch()
                    q_loss = agents[agt_name].update(train_data, idx)
                    Qnet_loss_list.append(q_loss)
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

        # * ---- adjest epsilon ----
        for agt_name in agent_name:
            agents[agt_name].epsilon = max(agents[agt_name].epsilon * 0.95, 0.2)

        # read best weights
        if replay_buffer.size > minimal_size:
            if episode_return > best_score:
                for agt_name in agent_name:
                    best_weight[agt_name] = agents[agt_name].q_net.state_dict()
                best_score = episode_return

            # load best weight
            if epi_training:
                for agt_name in agent_name:
                    agents[agt_name].q_net.load_state_dict(best_weight[agt_name])

        # save log to file and report train status
        evaluator.evaluate_and_save(writer, return_list, waiting_list, queue_list, speed_list,
                                    time_list, seed_list, ckpt_path, episode, agents, seed,
                                    critic_loss_list=Qnet_loss_list)

    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {(total_time / 60):.2f} min")

    return return_list, total_time // 60

def run_Fixed_agent(
    env: object,
    agent: object,
    agent_name: list,
    writer: int,
    total_episodes: int,
    seed: int,
    ckpt_path: str,
    evaluator: object,
    **kwargs,
):
    return_list = []
    waiting_list = []
    queue_list = []
    speed_list = []
    time_list = []
    seed_list = []
    start_time = time.time()

    for episode in range(total_episodes):
        episode_return = 0

        # * ---- execute simulation ----
        state, done, truncated = env.reset(seed=seed)[0], False, False
        simulation_time = 0
        state = torch.tensor(list(state.values()))
        while not (done | truncated):
            # for _ in trange(720, ncols=70):
            action = agent.take_action(simulation_time)
            next_state, reward, done, truncated, info = env.step(action)
            simulation_time += 5
            next_state = torch.tensor(list(next_state.values()))
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

        # save log to file and report train status
        evaluator.evaluate_and_save(writer, return_list, waiting_list, queue_list, speed_list,
                                    time_list, seed_list, ckpt_path, episode, None, seed)

    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {(total_time / 60):.2f} min")

    return return_list, total_time // 60