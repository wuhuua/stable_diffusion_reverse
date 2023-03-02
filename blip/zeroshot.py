import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from processor.dataPre import imgPre
import gc
# from accelerate import init_empty_weights

def generateContext(model_path,data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device is:"+device)
    gc.collect()
    if device=="cuda":
        print("using cuda to generate")
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.float16, device_map="auto")
    else:
        print("using cpu to generate")
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    images=imgPre(data_path)
    print(str(len(images))+"images are loaded to get prompts")
    text=[]
    for image in images:
        gc.collect()
        if(device=="cuda"):
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(images=image, return_tensors="pt")
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text.append(generated_text)
    print(str(len(text))+"texts are generated")
    return text