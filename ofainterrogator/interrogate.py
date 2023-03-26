import gc
from typing import List
import json
import math
import sys
import numpy as np
import torch
import tqdm
import clip
# 使用集成OFA的transformer模型请参考
# https://github.com/OFA-Sys/OFA/blob/feature/add_transformers/transformers.md
from transformers.models.ofa import OFATokenizer,OFAModel
from processor.dataPre import ImageGen, imgPrePath
from transformers.models.ofa.generate import sequence_generator

class ofai_config:
    def __init__(self,config_path,device="cuda" if torch.cuda.is_available() else "cpu"):
        '''
        加载config.json中的数据,读取两个模型的名称和地址,读取clip的字典
        '''
        with open(config_path,encoding = 'utf-8') as fp:
            data=json.load(fp)
            self.device=device
            self.ofa_path=data["ofa_path"]
            self.clip_model=data["clip_model"]
            self.clip_path=data["clip_path"]
            self.dicts=data["dicts"]
            self.chunk_size=data["chunk_size"]
            self.embedding_model=data["embedding_model"]
            self.embedding_dir=data["embedding_dir"]
            self.embedding_path=data["embedding_path"]
        fp.close()

class clipInterrogator:
    def __init__(self,config:ofai_config):

        self.device=config.device
        self.ofa_path=config.ofa_path
        self.clip_model=config.clip_model
        self.clip_path=config.clip_path
        self.dicts=config.dicts
        self.embedding_model=config.embedding_model
        self.embedding_dir=config.embedding_dir
        self.embedding_path=config.embedding_path

        print("using device:"+self.device)

        #初始化OFA模型
        self.ofa_model=OFA_model(self.ofa_path)

        # 初始化clip模型
        print("start initing clip model:"+self.clip_model)
        self.clip_model, self.clip_preprocess = clip.load(self.clip_path, self.device)
        print("clip initialized")

        # 初始化字典集
        print("start loading dicts")
        self.flavors = LabelTable(load_label(self.dicts, 'flavors.txt'), "flavors", self.clip_model, config)
        print("flavors loading OK")
        self.mediums = LabelTable(load_label(self.dicts, 'mediums.txt'), "mediums", self.clip_model, config)
        print("mediums loading OK")
        self.movements = LabelTable(load_label(self.dicts, 'movements.txt'), "movements", self.clip_model, config)
        print("movements loading OK")

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
            out = self.ofa_model.generate(self.inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=num_beams, no_repeat_ngram_size=2)
            out_captions = self.ofa_model.tokenizer.batch_decode(out, skip_special_tokens=True)
            out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
            embeddings = st_model.encode(out_captions).flatten()
            submissions.extend(embeddings)
        del self.ofa_model
        del self.clip_model
        del st_model
        torch.cuda.empty_cache()
        print(gc.collect())
        return submissions
    
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
            out = self.ofa_model.generate(self.inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=num_beams, no_repeat_ngram_size=2)
            out_captions = self.ofa_model.tokenizer.batch_decode(out, skip_special_tokens=True)
            out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
            text.extend(out_captions)
        return text

class OFA_model:
    def __init__(self,model_path):
        print("initializing models")
        self.tokenizer = OFATokenizer.from_pretrained(model_path)
        self.model = OFAModel.from_pretrained(model_path, use_cache=False).cuda()
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        txt = " what does the image describe?"
        self.inputs = self.tokenizer([txt], return_tensors="pt").input_ids
    

class LabelTable():
    def __init__(self, labels:List[str], desc:str, clip_model, config: ci_config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels

        # 初始化label数组
        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=False):
                text_tokens = clip.tokenize(chunk).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]
    
    # 单次相似度排序
    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1, reverse: bool=False) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
            if reverse:
                similarity = -similarity
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    # 分批次相似度排序
    def rank(self, image_features: torch.Tensor, top_count: int=1, reverse: bool=False) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count, reverse=reverse)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=False):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk, reverse=reverse)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def load_label(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items