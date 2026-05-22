import torch
import numpy as np
from util.tools import get_action

class Fiexed:
    ''' DQN算法 '''
    def __init__(self, flag, action_space, action_dict, agent_names):
        self.agent_names = agent_names
        self.action_dict = action_dict
        self.action_space = action_space
        self.flag = flag

    def take_action(self, current_time) -> dict:
        actions = get_action(current_time, self.flag, self.action_space, self.action_dict, self.agent_names)
        return actions