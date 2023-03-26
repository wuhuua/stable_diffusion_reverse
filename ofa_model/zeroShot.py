import sys
import torch
# 使用集成OFA的transformer模型请参考
# https://github.com/OFA-Sys/OFA/blob/feature/add_transformers/transformers.md
from transformers.models.ofa import OFATokenizer,OFAModel
from processor.dataPre import ImageGen, imgPrePath
from transformers.models.ofa.generate import sequence_generator

class OFA_model:
    def __init__(self,model_path):
        print("initializing models")
        self.tokenizer = OFATokenizer.from_pretrained(model_path)
        self.model = OFAModel.from_pretrained(model_path, use_cache=False).cuda()
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        txt = " what does the image describe?"
        self.inputs = self.tokenizer([txt], return_tensors="pt").input_ids

    def getCaption(self,data_path,batch_size=16,num_beams=1):
        images_path=imgPrePath(data_path)
        print(str(len(images_path))+" images are loaded to get prompts")
        text=[]
        imgen = ImageGen(data_path ,batch_size)
        sub_ids = []
        for b in imgen:
            for j in range(len(b[1])):
                sub_ids.extend([f"{b[1][j]}_{i}" for i in range(384)])
            img_batch = b[0]
            out = self.model.generate(self.inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=num_beams, no_repeat_ngram_size=2)
            out_captions = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
            text.extend(out_captions)
        return text
        
    def getEmbeddings(self,data_path,embedding_dir,embedding_path,batch_size=16,num_beams=1):
        
        # 初始化文本-嵌入编码模型
        print("embedding mode start")
        print("initing embedding model")
        sys.path.append(embedding_dir)
        from sentence_transformers import SentenceTransformer, models
        st_model = SentenceTransformer(embedding_path)
        print("embedding model initialized")

        images_path=imgPrePath(data_path)
        print(str(len(images_path))+"images are loaded to get prompts")
        submissions=[]
        imgen = ImageGen(data_path ,batch_size)
        sub_ids = []
        for b in imgen:
            for j in range(len(b[1])):
                sub_ids.extend([f"{b[1][j]}_{i}" for i in range(384)])
            img_batch = b[0]
            out = self.model.generate(self.inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=num_beams, no_repeat_ngram_size=2)
            out_captions = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
            embeddings = st_model.encode(out_captions).flatten()
            submissions.extend(embeddings)
        del self.tokenizer
        del self.model
        torch.cuda.empty_cache()
        return submissions
        