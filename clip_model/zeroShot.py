import os
import clip
import torch
from processor.dataPre import imgPre

def zeroShot(model_path,dataset_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_path, device)
    images=imgPre(dataset_path)
    problist=[]
    text_single=["a diagram", "a dog", "a cat"]
    for i in range(len(images)):
        image_input = preprocess(images[i]).unsqueeze(0).to(device)
        text_pre = clip.tokenize(text_single).to(device)
        with torch.no_grad():
            images_feature=model.encode_image(image_input)
            text_feature=model.encode_text(text_pre)
            logits_per_image, logits_per_text = model(image_input, text_pre)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            problist.append(probs)
    print("Label probs:", problist)