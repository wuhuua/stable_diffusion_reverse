from PIL import Image
from clip_interrogator import Config, Interrogator
import glob
from processor.dataPre import imgPre

def clip_interrogator(model_path,data_path):
    images=imgPre(data_path)
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    text=[]
    for image in images:
        text=ci.interrogate(image)
    return text

