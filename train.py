import argparse

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
    parser.add_argument('--save-folder', defualt='weights/', help='Directory for saving checkpoint models')
    opt=parser.parse_args()
    print(opt)