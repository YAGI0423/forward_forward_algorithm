import os
import tqdm
import argparse

import torch
from torch import optim
from torch.nn import Module

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

def load_model(model: Module, path: str) -> bool:
    if os.path.isfile(path):
        print(f'\tload model from {path}...')
        model.load_state_dict(torch.load(path))
        return True
    print(f'\tno model in {path}...')
    return False

def train(ff_model: Module, bp_model: Module, dataLoader: mnistDataLoader, device: str) -> None:
    ff_loss, bp_loss = list(), list()

    ff_model.train()
    bp_model.train()
    dataLoader = iter(dataLoader)
    for (x0, y0), (x1, y1) in zip(dataLoader, dataLoader):
        x0, y0 = x0.to(device), y0.to(device)
        x1, y1 = x1.to(device), y1.to(device)
        
        bp_loss.append(bp_model.update(x0, y0).item())
        bp_loss.append(bp_model.update(x1, y1).item())


        print(bp_loss)
        
        raise
        
    raise

def inference(ff_model: Module, bp_model: Module, dataLoader: mnistDataLoader, device: str) -> tuple[float, float]:
    ff_acc, bp_acc = list(), list()

    ff_model.eval()
    bp_model.eval()
    with torch.no_grad():
        for x, y in dataLoader:
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


    print(f'+ MODEL SHAPE\n\tFF-Model:  {args.ff_dims}\n\tBP-Model:  {args.bp_dims}', end='\n\n')
    ff_model = models.FFModel(dims=args.ff_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)
    bp_model = models.BPModel(dims=args.bp_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)

    print(f'+ Load Model')
    load_model(ff_model, FF_PATH)
    load_model(bp_model, BP_PATH)

    if args.mode == 'TRAIN':
        train_dataLoader = mnistDataLoader.get_loader(train=True, batch_size=args.train_batch_size)
        train(ff_model, bp_model, train_dataLoader, device=DEVICE)

    elif args.mode == 'INFERENCE':
        print(f'\n+ Accuracy on MNIST Test Set')
        test_dataLoader = mnistDataLoader.get_loader(train=False, batch_size=args.test_batch_size)
        ff_acc, bp_acc = inference(ff_model, bp_model, test_dataLoader, device=DEVICE)
        print(f'\tFF Model: {ff_acc:.3f}')
        print(f'\tBP Model: {bp_acc:.3f}')