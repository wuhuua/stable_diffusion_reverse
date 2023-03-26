import glob
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 使用直接加载的方式来加载文件
def imgPrePath(data_path):
    sample_images = glob.glob(data_path+"/images/*")
    return sample_images

def imgPre(data_path):
    sample_images = glob.glob(data_path+"/images/*")
    images=[]
    for i,path in enumerate(sample_images):
        images.append(Image.open(path).convert("RGB"))
    return images

# 使用图片加载器来自定义加载文件
class ImageGen(Dataset):
    def __init__(self, root, batch_size=32):
        self.root = root
        self.im_paths = os.listdir(self.root)
        self.batch_size = batch_size
        self.sz = len(self.im_paths)
        self.genlen = self.sz//self.batch_size + int(self.sz%self.batch_size > 0)
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        self.patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
        ])
    
    # 使用__getitem__方法获取类中的变量
    def __getitem__(self, index):
        if index >= self.genlen:
            raise IndexError("Out of bounds")
        
        l, r = index*self.batch_size, min(self.sz, (index+1)*self.batch_size)
        
        f_paths = glob.glob(self.root+"/images/*")
        f_ids = [self.im_paths[i][:-4] for i in range(l,r)]
        
        ims = [Image.open(f_path) for f_path in f_paths]
        ims = [self.patch_resize_transform(im).cuda().unsqueeze(0) for im in ims]
        ims = torch.cat(ims)
        
        return ims, f_ids
    
    def __len__(self):
        return self.genlen