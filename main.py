import torch
import clip
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from clip_model.zeroShot import zeroShot
from blip.zeroshot import generateContext
from blip.zeroshotembedding import generateEmbedding
from processor.dataPre import imgPre
# from clipinterrogator.Interrogate import clip_interrogator

def test_models():
    print("Torch version:", torch.__version__)
    print("clip models:",clip.available_models())

def execute(model_path,data_path,model,extend_path=None,model_extend_path=None,batch_size=None):
    test_models()
    if(model=="zeroshot"):
        zeroShot(model_path,data_path)
    if(model=="blip2"):
        return generateContext(model_path,data_path,batch_size)
    if(model=="test_data"):
        imgPre(data_path)
    if(model=="blip2-embedding"):
        return generateEmbedding(model_path,data_path,extend_path,model_extend_path,batch_size)
'''
    if(model=="clip-interrogator"):
        return clip_interrogator(model_path,data_path)
'''

if __name__=='__main__':
    # execute("./models/ViT-B-32.pt","./static","zeroshot")
    # execute("./models/blipModels/simple/","./static","blip2")
    execute("","./static","test_data")