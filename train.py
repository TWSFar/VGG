import argparse
import visdom
import time

from models import *
from data import *
from utils import *

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader


hyp = {'giou': .035,  # giou loss gain
       'xy': 0.20,  # xy loss gain
       'wh': 0.10,  # wh loss gain
       'cls': 0.035,  # cls loss gain
       'conf': 1.61,  # conf loss gain
       'conf_bpw': 3.53,  # conf BCELoss positive_weight
       'iou_t': 0.29,  # iou target-anchor training threshold
       'lr0': 0.001,  # initial learning rate
       'lr_gamma': 0.1, # learning decay factory
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


def train(mode='train'):
    
    # config parameter
    device = torch_utils.select_device()
    start_epoch = 0
    
    #visualization
    if opt.visdom:
        vis = visdom.Visdom()
        vis_title = 'VGG.Pytorch on DogCat'
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(vis, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(vis, 'Epoch', 'Loss', vis_title, vis_legend)
    
    # dataset load
    dataset = DogCat(opt.dataset_path, opt.img_size, mode)
    dataloader = DataLoader(
                            dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn
                            )

    # model and optimizer create , init and load checkpoint    
    model = VGG(opt.cfg, opt.img_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    if opt.resume:
        best = opt.save_folder + 'best.pt'
        latest = opt.save_folder + 'latest.pt'
        chkpt = torch.load(latest)
        model_dict = model.state_dict()
        pretrained_dict = chkpt['model']
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        
        best_loss = chkpt['best_loss']
        start_epoch = chkpt['epoch'] + 1
    else:
        model.module_list.apply(weights_init)
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=hyp['lr_gamma']), 
    scheduler.last_epoch = start_epoch - 1

    # train
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGG training with Pytorch')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--cfg', type=str, default='cfg/vgg16.cfg', help='cfg file path')
    parser.add_argument('--single-scale', action='store_true', help='train at fixed size')
    parser.add_argument('--img-size', type=int, default=224, help='inference size')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--number-classes', type=int, default=2, help='number of classes')
    parser.add_argument('--dataset-path', type=str, default='datasets/DogCat', help='dataset path')
    parser.add_argument('--save-folder', type=str, default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--accumulate', type=int, default=1, help='number of batches to accumulate before optimizing')
    parser.add_argument('--visdom', default=True, type=bool, help='Use visdom for loss visualization')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    opt=parser.parse_args()
    print(opt)

    train()