import sys
from typing import List
import gc
import json
import math
import os
from PIL import Image
import clip
import numpy as np
import torch
from tqdm import tqdm
from batchmaker.makebatch import make_batches
from processor.dataPre import imgPrePath
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class ci_config:
    def __init__(self,config_path,device="cuda" if torch.cuda.is_available() else "cpu"):
        '''
        加载config.json中的数据,读取两个模型的名称和地址,读取clip的字典
        '''
        with open(config_path,encoding = 'utf-8') as fp:
            data=json.load(fp)
            self.device=device
            self.blip_model=data["blip_model"]
            self.blip_path=data["blip_path"]
            self.clip_model=data["clip_model"]
            self.clip_path=data["clip_path"]
            self.dicts=data["dicts"]
            self.chunk_size=data["chunk_size"]
            self.embedding_model=data["embedding_model"]
            self.embedding_dir=data["embedding_dir"]
            self.embedding_path=data["embedding_path"]
        fp.close()

class clipInterrogator:
    def __init__(self,config:ci_config):

        self.device=config.device
        self.blip_model=config.blip_model
        self.blip_path=config.blip_path
        self.clip_model=config.clip_model
        self.clip_path=config.clip_path
        self.dicts=config.dicts
        self.embedding_model=config.embedding_model
        self.embedding_dir=config.embedding_dir
        self.embedding_path=config.embedding_path

        print("using device:"+self.device)

        # 初始化blip模型
        print("start initing blip model:"+self.blip_model)
        if(self.device=="cuda"):
            self.blip_processor = Blip2Processor.from_pretrained(self.blip_path,torch_dtype=torch.float16, device_map="auto")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(self.blip_path,torch_dtype=torch.float16, device_map="auto")
        else:
            self.blip_processor = Blip2Processor.from_pretrained(self.blip_path)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(self.blip_path)
        print("blip initialized")

        print(gc.collect())

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

    def delModel(self):
        del self.blip_processor
        del self.blip_model
        del self.clip_model
        del self.mediums
        del self.flavors
        del self.movements
        print(gc.collect())
        torch.cuda.empty_cache()

    # 图片特征提取
    def image_to_features(self, image: Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    # 图片-文本生成
    def getCaption(self,data_path,batch_size=16,max_flavors=3,num_beams=1):
        images_path=imgPrePath(data_path)
        prompts=[]
        for batch in make_batches(images_path, batch_size):
            images_batch=[]
            flaves=[]
            medium=[]
            for i,path in enumerate(batch):
                images_batch.append(Image.open(path).convert("RGB"))
            # 使用blip生成caption
            if(self.device=="cuda"):
                inputs = self.blip_processor(images=images_batch, return_tensors="pt").to(self.device, torch.float16)
            else:
                inputs = self.blip_processor(images=images_batch, return_tensors="pt")
            generated_ids = self.blip_model.generate(**inputs,max_length=20, min_length=5,num_beams=num_beams)
            generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            # 使用clip添加标签

            for i,image in enumerate(images_batch):
                image_feature=self.image_to_features(image)
                flaves=", ".join(self.flavors.rank(image_feature, max_flavors))
                medium=self.mediums.rank(image_feature, 1)[0]
                movement = self.movements.rank(image_feature, 1)[0]
                prompts.append(f"{generated_text[i][:-1]}, {medium}, {movement}, {flaves}")
        self.delModel()
        print(str(len(prompts))+"texts are generated")
        print("collecting rubbish:")
        print(gc.collect())
        return prompts
    
    # 图片-嵌入编码生成
    def getEmbeddings(self,data_path,batch_size=16,max_flavors=3,num_beams=1):

        # 初始化文本-嵌入编码模型
        print("embedding mode start")
        print("initing embedding model:"+self.embedding_model)
        sys.path.append(self.embedding_dir)
        from sentence_transformers import SentenceTransformer, models
        st_model = SentenceTransformer(self.embedding_path)
        print("embedding model initialized")

        images_path=imgPrePath(data_path)
        submissions=[]
        for batch in make_batches(images_path, batch_size):
            images_batch=[]
            flaves=[]
            medium=[]
            for i,path in enumerate(batch):
                images_batch.append(Image.open(path).convert("RGB"))
            # 使用blip生成caption
            if(self.device=="cuda"):
                inputs = self.blip_processor(images=images_batch, return_tensors="pt").to(self.device, torch.float16)
            else:
                inputs = self.blip_processor(images=images_batch, return_tensors="pt")
            generated_ids = self.blip_model.generate(**inputs,max_length=20, min_length=5, num_beams=num_beams)
            generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            # 使用clip添加标签
            prompts=[]
            for i,image in enumerate(images_batch):
                image_feature=self.image_to_features(image)
                flaves=", ".join(self.flavors.rank(image_feature, max_flavors))
                medium=self.mediums.rank(image_feature, 1)[0]
                prompts.append(f"{generated_text[i][:-1]}, {medium}, {flaves}")
            embeddings = st_model.encode(prompts).flatten()
            submissions.extend(embeddings)
        self.delModel()
        del st_model
        print(str(len(submissions))+"texts are generated")
        print("collecting rubbish:")
        print(gc.collect())
        return submissions 


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

