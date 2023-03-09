# CLIP version

***by Iscolito***

> 这是使用CLIP标签化的图片生成文本ai，使用BLIP2进行Caption的预测，并且使用CLIP进行标签化

本框架整体使用BLIP+CLIP的方式进行文本推测

组合方式为：

```python
prompt=f"{generated_text[i][:-1]}, {medium}, {movement}, {flaves}"
```

其中`medium`为对图片的词语概括，`movement`为对图片的风格预测，`flaves`为图片特征关键字，`generated_text`为通过BLIP模型生成的预测文本



## CLIP 实现

CLIP使用官方给出的神经网络推理方式进行相似度匹配：https://github.com/openai/CLIP

将字典全部读入后存储到内存中

`self.flavors = LabelTable(load_label(self.dicts, 'flavors.txt'),"flavors", self.clip_model, config)`

其中录入的文本以Label形式进行封装，包含了对文本通过相似度进行排序的算法：

```python
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
```



使用的字典集为：

`medium.txt`(共95行)

```
a 3D render
a black and white photo
a bronze sculpture
a cartoon
a cave painting
a character portrait
a charcoal drawing
a child's drawing
a color pencil sketch
a colorized photo
a comic book panel
a computer rendering
a cross stitch
...
```



`movement.txt`(共200行)

```
abstract art
abstract expressionism
abstract illusionism
academic art
action painting
aestheticism
afrofuturism
altermodern
american barbizon school
american impressionism
...
```



`flavor.txt`(共100970行)

flavor是标记图片的关键字典，通过大数据量来尽量完成对图片的可能文本特征的全覆盖。

```
highly detailed
sharp focus
intricate
digital painting
illustration
octane render
smooth
8 k
elegant
8k
cinematic
...
```

## BLIP实现

blip通过加载大模型的方式提高神经网络推理的准确率：

```python

self.blip_processor = Blip2Processor.from_pretrained(self.blip_path,torch_dtype=torch.float16, device_map="auto")
self.blip_model = Blip2ForConditionalGeneration.from_pretrained(self.blip_path,torch_dtype=torch.float16, device_map="auto")
```

为了避免模型过大导致的内存溢出，使用半精度浮点数的方式加载模型并进行神经网络推理，通过内存读取后保存在GPU上，因此注意在模型加载顺序上保证BLIP模型的加载空间。



## 多批次训练机制



```python
def make_batches(l, batch_size=16):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]
```

当输入数据量过大时，使用多批次训练机制来完成训练。



## 接口介绍和配置文件

`main.py`是框架的唯一接口，其中定义的函数参数为：

| 参数              | 默认值 |
| ----------------- | ------ |
| model_path        | None   |
| data_path         | None   |
| model             | None   |
| extend_path       | None   |
| model_extend_path | None   |
| batch_size        | 16     |
| max_flavors       | 3      |
| num_beams         | 1      |

- model_path:模型的地址
- data_path:数据的地址
- model:模型
- extend_path:配置文件地址等
- model_extend_path:模型的扩展地址
- batch_size:每个推理批次的大小
- max_flavors:flavor的标签数量
- num_beams:一个beam_search的搜索量



配置文件内容为:

```json
{
    "blip_model":"blip227b",
    "blip_path":"./models/stable-diffusion-image-to-prompts",
    "clip_model":"ViT-B-32",
    "clip_path":"./models/ViT-B-32.pt",
    "dicts":"./dataset/dicts/",
    "chunk_size":2048,
    "embedding_model":"all-MiniLM-L6-v2",
    "embedding_dir":"./models/sentence-transformers-222/sentence-transformers",
    "embedding_path":"./models/sentence-transformers-222/sentence-transformers/all-MiniLM-L6-v2"
}
```

如在kaggle上可以通过下列方式来进行调用：

```python
import sys
sys.path.append('../input')    
sys.path.append('../input/clipv1/clip_version') 
print(sys.path)

from clipv1.clip_version.main import execute
prompt_embeddings=execute(
    model="clip-interrogator-embedding",
    data_path="/kaggle/input/stable-diffusion-image-to-prompts",
    extend_path="/kaggle/input/reversesdconfig/kaggle_config.json",
    batch_size=16,
    max_flavors=3,
    num_beams=6
)
```

