import glob
from PIL import Image

def imgPre(data_path):
    sample_images = glob.glob(data_path+"/images/*")[:]
    print(sample_images)
    images=[]
    for i,impath in enumerate(sample_images):
        image = Image.open(impath)
        images.append(image)
    return images