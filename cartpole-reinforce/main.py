import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
