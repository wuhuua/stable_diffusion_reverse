'''use it only when in kaggle
! pip install ftfy regex tqdm
! pip install git+https://github.com/openai/CLIP.git
'''

import torch
import clip
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from zeroshot.zeroShot import zeroShot
from blip.zeroshot import generateContext
from processor.dataPre import imgPre

def test_models():
    print("Torch version:", torch.__version__)
    print("clip models:",clip.available_models())

def execute(model_path,data_path,model):
    test_models()
    if(model=="zeroshot"):
        zeroShot(model_path,data_path)
    if(model=="blip2"):
        generateContext(model_path,data_path)
    if(model=="test_data"):
        imgPre(data_path)

if __name__=='__main__':
    # execute("./models/ViT-B-32.pt","./static","zeroshot")
    execute("./models/blipModels/simple/","./static","blip2")
    # execute("","./static","test_data")