import argparse
from tqdm import tqdm

from utils import *
from models import *
from data import *

import torch
from torch.utils.data import DataLoader


def test(
    opt,
    img_size = 224,
    model=None,
    mode='test',
    classes=['Cat', 'Dog']):
    
    # Configure
    device, gpu_num = torch_utils.select_device()
    best = osp.join(opt.save_folder, 'best.pt')
    latest = osp.join(opt.save_folder, 'latest.pt')
    used_gpu = False

    # model
    if model is None:
        temp = 'weights/vgg16.pt'
        model = VGG(opt.cfg, img_size)
        chkpt = torch.load(temp)
        model_dict = model.state_dict()
        pretrained_dict = chkpt['model']
        # new_dict = {}
        # for k, v in pretrained_dict.items():
        #     for k2 in model_dict.keys():
        #         if k in k2 and 'conv' in k2:
        #             new_dict[k2] = v  
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and 'fc' not in k}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(temp)['model']) 
    
    # gpu set
    if opt.gpu > 1 and gpu_num > 1:
        device_id = []
        for i in range(min(opt.gpu, gpu_num)):
            device_id.append(i)
        model = torch.nn.DataParallel(model, device_ids=device_id)
        model.to(device) 
        used_gpu = True
    else:
        model.to(device)
    
    # Loss
    criterion = nn.CrossEntropyLoss().to(device)
    
    # dataset
    dataset = DogCat(opt.testset_path, img_size, mode)\
    
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # test
    model.eval()
    loss, p, r, f1 = 0, 0, 0, 0
    total = 0
    corrects = 0
    stats = []
    for batch_i, (imgs, labels, file_) in enumerate(tqdm(dataloader, desc='Computing Loss')):
        labels = labels.to(device)
        imgs = imgs.to(device)
        labels = labels[:, 1].view(-1)
        # Run model
        output = model(imgs)

        # Compute Loss
        loss_i = criterion(output, labels)
        loss = loss + loss_i.item()


        _, predicted = output.max(1)
        corrects += predicted.eq(labels).sum().item()
        
        labels = labels.tolist()
        total += len(labels)
        
        for x, y in zip(predicted.tolist(), labels):
            stats.append((x, y))

    loss = loss / total
    stats = np.array(stats).astype(np.int64)
    num_per_class = np.bincount(stats[:, 1], minlength=opt.number_classes)

    corrects = 1. * corrects / total
    if len(stats):
        p, r, f1 = res_per_class(stats, total, opt.number_classes)
        mp, mr, mf1 = p.mean(), r.mean(), f1.mean()    
    
    print(('%10s' * 6) % ('Class_num', 'Labels', 'P', 'R', 'F1', 'loss'))
    print(("%10d"+"%10.3g"*5) % (opt.number_classes, total, corrects, mr, mf1, loss))

    if opt.number_classes > 1 and len(stats):
        for i, c in enumerate(classes):
            print(("%10s"+"%10d" + "%10.3g"*3) % (c, num_per_class[i], p[i], r[i], f1[i]))
    print()

    return corrects, loss, mf1 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/vgg16.cfg', help='cfg file path')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size')
    parser.add_argument('--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--number-classes', type=int, default=2, help='number of classes')
    parser.add_argument('--testset-path', type=str, default='datasets/DogCat/test', help='path of dataset')
    parser.add_argument('--save-folder', type=str, default='weights', help='Directory for saving checkpoint models')
    parser.add_argument('--gpu', default=4, type=int, help='number of gpu')
    opt = parser.parse_args()
    
    with torch.no_grad():
        result = test(opt, mode='test')

