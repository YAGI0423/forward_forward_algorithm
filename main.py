import os
import argparse

import torch
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

def inference(model, dataLoader, device: str):
    model.eval()
    with torch.no_grad():
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            
            y_hat = model.inference(x)
            
            raise

if __name__ == '__main__':
    args = get_args()
    FF_PATH = './model/ff_model.pk'
    BP_PATH = './model/bp_model.pk'
    FIGURE_PATH = './figures/'
    DEVICE = args.device.lower()

    print(f'MODEL SHAPE: {args.ff_dims}, {args.bp_dims}')

    ff_model = models.FFModel(dims=args.ff_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)
    bp_model = models.BPModel(dims=args.bp_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)
    loss_function = torch.nn.MSELoss()
    

    if args.mode == 'INFERENCE': #INFERENCE
        if os.path.isfile(FF_PATH):
            ff_model.load_state_dict(torch.load(FF_PATH))

        if os.path.isfile(BP_PATH):
            bp_model.load_state_dict(torch.load(BP_PATH))

        test_dataLoader = mnistDataLoader.get_loader(train=False, batch_size=args.test_batch_size)

        ff_model.eval()
        bp_model.eval()
        with torch.no_grad():
            for x, y in test_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                ff_y_hat = ff_model.inference(x).argmax(dim=1)
                bp_y_hat = bp_model.inference(x).argmax(dim=1)

                

        inference(ff_model, test_dataLoader, device=DEVICE)
        raise

    elif args.mode == 'TRAIN': #TRAIN
        print('train')