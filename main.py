import os
import argparse

import torch
from torch import optim

import ffmodel
from util import mnistDataLoader, models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='INFERENCE', choices=('INFERENCE', 'TRAIN'))
    parser.add_argument('--dims', type=int, default=[28*28, 100], nargs='+')
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

def inference(model, dataLoader, loss_fc, device: str):
    model.eval()
    with torch.no_grad():
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            

            raise

if __name__ == '__main__':
    args = get_args()
    FF_PATH = './model/ff_model.pk'
    BP_PATH = './model/bp_model.pk'
    FIGURE_PATH = './figures/'
    DEVICE = args.device.lower()

    print(f'MODEL SHAPE: {args.dims}')

    ff_model = ffmodel.FFModel(dims=args.dims, optimizer=get_optim(args), lr=args.lr).to(DEVICE)
    bp_model = models.BPModel(dims=args.dims, optimizer=get_optim(args), lr=args.lr).to(DEVICE)
    loss_function = torch.nn.MSELoss()
    

    if args.mode == 'INFERENCE': #INFERENCE
        if os.path.isfile(FF_PATH):
            ff_model.load_state_dict(torch.load(FF_PATH))

        if os.path.isfile(BP_PATH):
            bp_model.load_state_dict(torch.load(BP_PATH))

        test_dataLoader = mnistDataLoader.get_loader(train=False, batch_size=args.test_batch_size)

        inference(bp_model, test_dataLoader, loss_fc=loss_function, device=DEVICE)
        raise

    elif args.mode == 'TRAIN': #TRAIN
        print('train')