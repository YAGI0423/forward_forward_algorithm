import os
import argparse

import torch
from torch import nn
from torch import optim

from util import mnistDataLoader, models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='INFERENCE', choices=('INFERENCE', 'TRAIN'))
    parser.add_argument('--ff_dims', type=int, default=[28*28, 100, 10], nargs='+')
    parser.add_argument('--bp_dims', type=int, default=[28*28, 100, 10], nargs='+')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=('SGD', 'ADAM'))
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='CPU', choices=('CPU', 'CUDA'))
    return parser.parse_args()

def get_optim(args) -> optim:
    if args.optimizer == 'SGD':
        return optim.SGD
    return optim.Adam

def load_model(model: nn.Module, path: str) -> bool:
    if os.path.isfile(path):
        print(f'load model from {path}...')
        model.load_state_dict(torch.load(path))
        return True
    print(f'no model in {path}...')
    return False

def inference(ff_model, bp_model, dataLoader, device: str) -> tuple[list, list]:
    ff_acc, bp_acc = list(), list()

    ff_model.eval()
    bp_model.eval()
    with torch.no_grad():
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)

            ff_y_hat = ff_model.inference(x).argmax(dim=1)
            bp_y_hat = bp_model.inference(x).argmax(dim=1)
    return ff_acc, bp_acc


if __name__ == '__main__':
    args = get_args()
    FF_PATH = './model/ff_model.pk'
    BP_PATH = './model/bp_model.pk'
    FIGURE_PATH = './figures/'
    DEVICE = args.device.lower()


    print(f'+ MODEL SHAPE\n\tFF-Model:  {args.ff_dims}\n\tBP-Model:  {args.bp_dims}', end='\n\n')
    ff_model = models.FFModel(dims=args.ff_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)
    bp_model = models.BPModel(dims=args.bp_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)

    
    if args.mode == 'INFERENCE': #INFERENCE
        load_model(ff_model, FF_PATH)
        load_model(bp_model, BP_PATH)

        test_dataLoader = mnistDataLoader.get_loader(train=False, batch_size=args.test_batch_size)
        ff_acc, bp_acc = inference(ff_model, bp_model, test_dataLoader, device=DEVICE)
        print(ff_acc, bp_acc)
        raise

    elif args.mode == 'TRAIN': #TRAIN
        print('train')