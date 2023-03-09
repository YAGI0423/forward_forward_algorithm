import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import Tensor
from torch import optim
from torch.nn import Module

from util import mnistDataLoader, models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='INFERENCE', choices=('INFERENCE', 'TRAIN'))
    parser.add_argument('--ff_dims', type=int, default=[28*28, 100, 10], nargs='+')
    parser.add_argument('--bp_dims', type=int, default=[28*28, 100, 10], nargs='+')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=('SGD', 'ADAM'))
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='CPU', choices=('CPU', 'CUDA'))
    return parser.parse_args()

def print_set_info(args: argparse.Namespace) -> None:
    print('\n\n')
    print(f'SETTING INFO'.center(60, '='))
    print(f'+ Mode: {args.mode}({args.device})')
    print(f'+ Epoch: {args.epoch}', end='\n\n')
    print(f'+ Optimizer: {args.optimizer}(lr={args.lr:.3f})', end='\n\n')
    print(f'+ Batch size\n\t* Train: {args.train_batch_size}\n\t* Test: {args.test_batch_size}', end='\n\n')
    print(f'+ Model shape\n\t* FF-Model:  {args.ff_dims}\n\t* BP-Model:  {args.bp_dims}', end='\n\n')

def get_optim(args: argparse.Namespace) -> optim:
    if args.optimizer == 'SGD':
        return optim.SGD
    return optim.Adam

def load_model(model: Module, path: str) -> bool:
    if os.path.isfile(path):
        print(f'\tload model from {path}...')
        model.load_state_dict(torch.load(path))
        return True
    print(f'\tno model in {path}...')
    return False

def get_neg_y(y: Tensor, class_num: int=10) -> Tensor:
    '''
    return random label 'negative y', except real label
    + input shape: (Batch, )
    + output shape: (Batch, )
    '''
    device = y.device
    batch_size = y.size(0)
    
    able_idxs = torch.arange(class_num).unsqueeze(0).repeat(batch_size, 1).to(device)
    able_idxs = able_idxs[able_idxs != y.view(batch_size, 1)].view(batch_size, class_num-1)
    
    rand_idxs = torch.randint(class_num - 1, size=(batch_size, ))
    return able_idxs[range(batch_size), rand_idxs]

def train(ff_model: Module, bp_model: Module, dataLoader: mnistDataLoader, device: str) -> None:
    dataset_size = dataLoader.__len__()

    ff_model.train()
    bp_model.train()
    loaderIter = iter(dataLoader)
    for (x0, y0), (x1, y1) in tqdm(zip(loaderIter, loaderIter), total=dataset_size//2):
        #x0, y0 is for positive data
        #x1, y1 is for negative data
        x0, y0 = x0.to(device), y0.to(device)
        x1, y1 = x1.to(device), y1.to(device)
        
        bp_model.update(x0, y0)
        bp_model.update(x1, y1)
        ff_model.update(
            pos_x=x0, pos_y=y0, #positive data
            neg_x=x1, neg_y=get_neg_y(y1), #negative data
        )
    
def inference(ff_model: Module, bp_model: Module, dataLoader: mnistDataLoader, device: str) -> tuple[float, float]:
    ff_acc, bp_acc = list(), list()

    ff_model.eval()
    bp_model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataLoader):
            x, y = x.to(device), y.to(device)

            ff_y_hat = ff_model.inference(x).argmax(dim=1)
            bp_y_hat = bp_model.inference(x).argmax(dim=1)

            ff_acc.extend(ff_y_hat.eq(y).float().tolist())
            bp_acc.extend(bp_y_hat.eq(y).float().tolist())
    return sum(ff_acc) / len(ff_acc), sum(bp_acc) / len(bp_acc)


if __name__ == '__main__':
    args = get_args()
    FF_PATH = './model/ff_model.pk'
    BP_PATH = './model/bp_model.pk'
    FIGURE_PATH = './figures/'
    DEVICE = args.device.lower()


    print_set_info(args)
    ff_model = models.FFModel(dims=args.ff_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)
    bp_model = models.BPModel(dims=args.bp_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)

    print(f'+ Load Model')
    load_model(ff_model, FF_PATH)
    load_model(bp_model, BP_PATH)
    print(f'=' * 60)

    ff_acc, bp_acc = None, None
    if args.mode == 'TRAIN':
        ff_acces, bp_acces = list(), list()

        print('\n\n')
        print(f'TRAIN MODEL'.center(60, '='))
        for _ in range(args.epoch):
            train_dataLoader = mnistDataLoader.get_loader(train=True, batch_size=args.train_batch_size)
            train(ff_model, bp_model, train_dataLoader, device=DEVICE)

            test_dataLoader = mnistDataLoader.get_loader(train=False, batch_size=args.test_batch_size)
            ff_acc, bp_acc = inference(ff_model, bp_model, test_dataLoader, device=DEVICE)
            ff_acces.append(ff_acc)
            bp_acces.append(bp_acc)
        
        #save figure of accuracy
        print('=' * 60)
        plt.subplot(1, 2, 1)
        plt.plot(ff_acces)
        plt.subplot(1, 2, 2)
        plt.plot(bp_acces)
        plt.show()

        #save model

    elif args.mode == 'INFERENCE':
        print('\n\n')
        print(f'INFERENCE'.center(60, '='))
        test_dataLoader = mnistDataLoader.get_loader(train=False, batch_size=args.test_batch_size)
        ff_acc, bp_acc = inference(ff_model, bp_model, test_dataLoader, device=DEVICE)
        print('=' * 60)


    #common code of Train & Inference
    print('\n\n')
    print(f'INFERENCE RESULT'.center(60, '='))
    print(f'+ Accuracy on MNIST Test Set')
    print(f'\tFF Model: {ff_acc:.3f}')
    print(f'\tBP Model: {bp_acc:.3f}')
    print('=' * 60)