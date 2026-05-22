import torch
import sys
import os
import pandas as pd
import time

class Evaluator:
    def __init__(self):

        self.episode = 0
        self.time_flag = time.time()
        print(f"\n| `seed`: Random seed of algorithm."
              f"\n| `episode`: Number of current episode."
              f"\n| `time`: Time spent (minute) from the start of training to this moment."
              f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| `VaeLoss`: VAE loss value."
              f"\n| {'seed':>3}  {'episode':>5}  {'time':>5}  |  {'avgR':>7}  {'stdR':>9}     | {'objC':>8}  {'objA':>8}  |  {'VaeLoss':>8}  ")

    def evaluate_and_save(self, writer, return_list, waiting_list, queue_list, speed_list, time_list,
                          seed_list, ckpt_path, episode, agent, seed,
                          actor_loss_list=False, critic_loss_list=False, **kwargs):
        vae_loss_list = kwargs.get('vae_loss_list', None)

        # * ---- 整理信息 ----
        system_type = sys.platform
        # ckpt/SAC
        mission_name = ckpt_path.split('/')[1]
        alg_name = ckpt_path.split('/')[2]
        # data/plot_data/highway/SAC/
        file_path = f"data/plot_data/{mission_name}/{alg_name}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        log_file_path = f"{file_path}/{seed}_{alg_name}_{system_type}.csv"
        ckpt_file_path = f"{ckpt_path}/{seed}_{alg_name}_{system_type}.pt"

        # * ---- 存储信息 ----
        return_save = pd.DataFrame()
        return_save["Algorithm"] = [alg_name] * len(return_list)
        return_save["Seed"] = seed_list
        return_save["Return"] = return_list
        return_save['waiting_list'] = waiting_list
        return_save['queue_list'] = queue_list
        return_save['speed_list'] = speed_list
        return_save['Actor loss'] = sum(actor_loss_list) / len(actor_loss_list) if actor_loss_list else None
        return_save['Critic loss'] = sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else None
        return_save['VAE loss'] = vae_loss_list if actor_loss_list else None
        return_save["Log time"] = time_list
        if writer > 0:
            if agent is not None:
                save_dict = {"agent": agent}
                if kwargs.get('vae', None):
                    save_dict['vae'] = kwargs.get('vae')
                torch.save(save_dict, ckpt_file_path)
            return_save.to_csv(log_file_path, index=False, encoding='utf-8-sig')

        # * ---- 打印信息 ----
        self.episode = episode + 1
        used_time = (time.time()- self.time_flag) / 60
        avg_r = return_save["Return"].mean()
        avg_w = return_save['waiting_list'].mean()
        std_r = return_save["Return"].std()
        actor_loss = return_save['Actor loss'].mean()
        critic_loss = return_save['Critic loss'].mean()
        vae_loss = return_save['VAE loss'].mean()
        print(f"| {seed:3d}  {self.episode:5d}    {used_time:5.2f}   "
              f"| {avg_r:9.2f}  {std_r:9.2f}    "
              f"| {critic_loss:8.1f}  {actor_loss:8.1f}  "
              f"| {vae_loss:8.1f}   ")