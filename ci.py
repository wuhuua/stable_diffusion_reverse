import torch
import clip
from clip_model.zeroShot import zeroShot
from blip.zeroshot import generateContext
from blip.zeroshotembedding import generateEmbedding
from processor.dataPre import imgPre
from clipinterrogator.Interrogate import clipInterrogator,ci_config
from clipinterrogator.Interrogate import load_label

def test_models():
    print("Torch version:", torch.__version__)
    print("clip models:",clip.available_models())

'''
execute工具是神经网络推理的主工具,通过传入参数来调控模型和数据的具体路径,
也可以在extend_path中传入配置文件的路径来自定义
'''

def execute(
        model_path=None,
        data_path=None,
        model=None,
        extend_path=None,
        model_extend_path=None,
        batch_size=16,
        max_flavors=3,
        num_beams=1
        ):
    
    test_models()
    if(model=="clip_model"):
        zeroShot(model_path,data_path)
    if(model=="blip2"):
        return generateContext(model_path,data_path,batch_size)
    if(model=="test_data"):
        imgPre(data_path)
    if(model=="test_dict"):
        return load_label("./dataset/dicts/",'mediums.txt')
    if(model=="blip2-embedding"):
        return generateEmbedding(model_path,data_path,extend_path,model_extend_path,batch_size)
    if(model=="clip-interrogator"):
        ci=clipInterrogator(ci_config(extend_path))
        return ci.getCaption(data_path,batch_size,max_flavors,num_beams)
    if(model=="clip-interrogator-embedding"):
        ci=clipInterrogator(ci_config(extend_path))
        res=ci.getEmbeddings(data_path,batch_size,max_flavors,num_beams)
        del ci
        torch.cuda.empty_cache()
        return res

if __name__=='__main__':
    # execute("./models/ViT-B-32.pt","./static","clip_model")
    print(execute(model="test_dict"))
    # execute("./models/blipModels/simple/","./static","blip2")
    # execute("","./static","test_data")