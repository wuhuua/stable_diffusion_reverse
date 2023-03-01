'''use it only when in kaggle
! pip install ftfy regex tqdm
! pip install git+https://github.com/openai/CLIP.git
'''

import torch
import clip
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset

print("Torch version:", torch.__version__)
