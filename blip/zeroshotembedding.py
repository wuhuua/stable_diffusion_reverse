import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from processor.dataPre import imgPrePath
import gc
from PIL import Image
import sys
from batchmaker.makebatch import make_batches

def generateEmbedding(model_path,data_path,extend_path,model_extend_path,batch_size=16):
    sys.path.append(extend_path)
    from sentence_transformers import SentenceTransformer, models
    st_model = SentenceTransformer(model_extend_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device is:"+device)
    print("collecting rubbish:")
    print(gc.collect())
    if device=="cuda":
        print("using cuda to generate")
        processor = Blip2Processor.from_pretrained(model_path,torch_dtype=torch.float16, device_map="auto")
        model = Blip2ForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.float16, device_map="auto")
    else:
        print("using cpu to generate")
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    images_path=imgPrePath(data_path)
    print(str(len(images_path))+"images are loaded to get prompts")
    gc.collect()
    submissions=[]
    for batch in make_batches(images_path, batch_size):
        images_batch=[]
        for i,path in enumerate(batch):
            images_batch.append(Image.open(path).convert("RGB"))
        if(device=="cuda"):
            inputs = processor(images=images_batch, return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(images=images_batch, return_tensors="pt")
        generated_ids = model.generate(**inputs,max_length=20, min_length=5)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        embeddings = st_model.encode(generated_text).flatten()
        submissions.extend(embeddings)
    del processor
    del model
    print("collecting rubbish:")
    print(gc.collect())
    return submissions