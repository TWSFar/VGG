import cv2

import torch
from torch import nn
import torch.nn.functional as F

from .parse_config import *
from collections import OrderedDict


def create_modules(module_defs):
    """
    The method use to create modle, used by __init__ of module which extent nn.Module.

    module_defs is a dict data, presente struct of your module, the return is a ModuleList.

    """
    hyperparams = module_defs.pop(0)
    out_channels = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stirde = int(module_def['stride'])
            pad = int(module_def['pad'])
            modules.add_module('conv_{}'.format(i), nn.Conv2d(in_channels=out_channels[-1],
                                                out_channels=filters,
                                                kernel_size=kernel_size,
                                                stride=stirde,
                                                padding=pad,
                                                bias=not bn))
            if bn:
                modules.add_module('batch_norm_{}'.format(i), nn.BatchNorm2d(filters))
            
        elif module_def['type'] == 'relu':
            if module_def['activation']:
                modules.add_module("leaky_{}".format(i), nn.LeakyReLU(0.1, inplace=True))
            else:
                modules.add_module('relu_{}'.format(i), nn.ReLU(inplace=True))

        elif module_def['type'] == 'maxpooling':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            modules.add_module('maxpooling_{}'.format(i), nn.MaxPool2d(kernel_size=kernel_size,
                                                                       stride=stride))

        module_list.append(modules)
        out_channels.append(filters)
    return module_list


class VGG(nn.Module):
    '''  
    VGGnet classes module
    '''
    def __init__(self, cfg, img_size):
        super(VGG, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.img_size = img_size
        self.module_defs[0]['height'] = img_size
        self.module_list = create_modules(self.module_defs)

        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
     

    def forward(self, x):
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.1))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.1))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    best = 'G:\\CV\\Reading\\VGG\\weights\\best.pt'
    model = VGG("cfg/vgg16.cfg", 224)
    model.load_state_dict(torch.load(best)['model']) 
    # img = cv2.imread("data/001.jpg")
    # height, width, channels = img.shape()
    # imgs = torch.from_numpy(img).permute(2, 0, 1)     
    input = torch.autograd.Variable(torch.randn(5, 3, 224, 224))
    res = model(input)
    print(res)