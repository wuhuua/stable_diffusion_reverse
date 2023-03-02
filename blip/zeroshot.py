import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from processor.dataPre import imgPre

def generateContext(model_path,data_path):
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    images=imgPre(data_path)
    print(images)
    text=[]
    for i in range(len(images)):
        inputs = processor(images=images[i], return_tensors="pt") # .to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text.append(generated_text)
    print(text)