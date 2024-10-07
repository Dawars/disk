import torch
import numpy as np

from disk import MatchDistribution, Features, NpArray, Image

class Reinforce:
    def __init__(self, reward, lm_kp):
        self.reward = reward
        self.lm_kp   = lm_kp
