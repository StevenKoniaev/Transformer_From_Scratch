import time
from collections import defaultdict
import mini_gpt_final
import torch
from torch.utils.data.dataloader import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

