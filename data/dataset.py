import os
import os.path as osp
from PIL import Image
import numpy as np
import cv2
import torch
from .transforms import *
from torch.utils import data


class DogCat(data.Dataset):
    def __init__(self, root, img_size=224, mode='train'):
        imgs = os.listdir(root)
        self.img_files = [os.path.join(root, img) for img in imgs]
        self.img_size = img_size
        self.mode = mode


    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
    
        if self.mode == 'train':
            img = letterbox(img, self.img_size, self.mode) # resize image
        else:
            img = letterbox(img, self.img_size, self.mode)
            
        # show_image(img)    
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = normalize(img)
        img = torch.from_numpy(img).float() 
        label = np.array(1 if 'dog' in img_path.split('/')[-1] else 0)

        label_out = torch.zeros((1, 2))
        label_out[0, 1] = torch.from_numpy(label)
        return (img, label_out, img_path)


    @staticmethod
    def collate_fn(batch):
        img, label, img_path= list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0).long(), img_path
    

    def __len__(self):
        return len(self.img_files)


def show_image(img):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.show()



# dataset = DogCat('datasets/DogCat/train')

# for img, label in dataset:
#     print(img.size(), label)

