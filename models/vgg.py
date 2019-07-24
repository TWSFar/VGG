import cv2
import math
import torch
from torch import nn
import torch.nn.functional as F
try:
    from .parse_config import *
except:
    from parse_config import *
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
        self.fc1 = nn.Linear(512*7*7, 4096, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 2)
        
        for m in self.module_list.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                try:
                    m.bias.data.zero_()
                except:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            x = module(x)
        x = x.view(x.size(0), -1)
        x = self.batch_norm1(F.leaky_relu(self.fc1(x), 0.1))
        x = self.batch_norm2(F.leaky_relu(self.fc2(x), 0.1))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

def vgg16(pretrained=True, img_size=224):
    model = VGG(cfg='cfg/vgg16.cfg', img_size=224)
    if pretrained:
        chkpt = torch.load("/home/twsf/.cache/torch/checkpoints/vgg16_reducedfc.pth")
        model_dict = model.state_dict()
        new_dict = {}
        for k1, v1 in chkpt.items():
            for k2, v2 in model_dict.items():
                if k1 in k2 and v1.shape == v2.shape:
                    new_dict[k2] = v1
        # new_dict = {k: v for k, v in chkpt.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    
    return model

if __name__ == "__main__":
    model = vgg16()
    input = torch.autograd.Variable(torch.randn(5, 3, 224, 224))
    res = model(input)
    print(res)