import glob
from PIL import Image

def imgPrePath(data_path):
    sample_images = glob.glob(data_path+"/images/*")
    return sample_images

def imgPre(data_path):
    sample_images = glob.glob(data_path+"/images/*")
    images=[]
    for i,path in enumerate(sample_images):
        images.append(Image.open(path).convert("RGB"))
    return images