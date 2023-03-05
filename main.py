import os
import argparse

import ffmodel
from util import models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='INFERENCE', choices=('TRAIN', 'INFERENCE'))
    parser.add_argument('--dims', type=int, default=[28*28, 50], nargs='+')
    parser.add_argument('--epoch', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(f'MODEL SHAPE: {args.dims}')

    ff_model = ffmodel.FFModel(dims=args.dims)
    bp_model = models.BPModel(dims=args.dims)

    if args.mode == 'INFERENCE': #INFERENCE
        if os.path.isfile('./model/ff_model.py'):
            pass
        
        if os.path.isfile('./model/bp_model.py'):
            pass
    else: #TRAIN
        print('train')