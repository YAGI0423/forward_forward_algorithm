import os
import argparse
from tqdm import tqdm

import torch
from torch import Tensor
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader

from utils import utils
from models import models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='INFERENCE', choices=('INFERENCE', 'TRAIN'))
    parser.add_argument('--ff_dims', type=int, default=[28*28, 100, 10], nargs='+')
    parser.add_argument('--bp_dims', type=int, default=[28*28, 100, 10], nargs='+')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=('SGD', 'ADAM'))
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='CUDA', choices=('CPU', 'CUDA'))
    parser.add_argument('--seed', type=int, default=23)
    return parser.parse_args()

def seed_everything(seed: int) -> None:
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

def train(ff_model: Module, bp_model: Module, dataLoader: DataLoader, device: str) -> None:
    dataset_size = dataLoader.__len__()

    ff_model.train()
    bp_model.train()
    loaderIter = iter(dataLoader)
    tqdm_loader = tqdm(zip(loaderIter, loaderIter), total=dataset_size//2, desc='Train')
    for (x0, y0), (x1, y1) in tqdm_loader:
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
    
def inference(ff_model: Module, bp_model: Module, dataLoader: DataLoader, device: str) -> tuple[float, float]:
    get_mean = lambda x: sum(x) / len(x)
    ff_acces, bp_acces = list(), list()

    ff_model.eval()
    bp_model.eval()
    tqdm_loader = tqdm(dataLoader)
    with torch.no_grad():
        for x, y in tqdm_loader:
            x, y = x.to(device), y.to(device)

            ff_y_hat = ff_model.inference(x).argmax(dim=1)
            bp_y_hat = bp_model.inference(x).argmax(dim=1)

            ff_acc = ff_y_hat.eq(y).float().tolist()
            bp_acc = bp_y_hat.eq(y).float().tolist()

            ff_acces.extend(ff_acc)
            bp_acces.extend(bp_acc)

            tqdm_loader.set_description(f'ff ACC({get_mean(ff_acces):.3f}), bp ACC({get_mean(bp_acces):.3f})')
    return get_mean(ff_acces), get_mean(bp_acces)


if __name__ == '__main__':
    MODEL_HOME = './trained_model'
    FF_PATH = os.path.join(MODEL_HOME, 'ff_model.pk')
    BP_PATH = os.path.join(MODEL_HOME, 'bp_model.pk')
    FIGURE_HOME = './figures'
    args = get_args()
    DEVICE = args.device.lower()

    seed_everything(args.seed)
    print_set_info(args)
    ff_model = models.FFModel(dims=args.ff_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)
    bp_model = models.BPModel(dims=args.bp_dims, optimizer=get_optim(args), lr=args.lr, device=DEVICE)

    print(f'+ Load Model')
    if not os.path.isdir(MODEL_HOME):
        os.makedirs(MODEL_HOME)
    load_model(ff_model, FF_PATH)
    load_model(bp_model, BP_PATH)
    print(f'=' * 60)

    ff_acc, bp_acc = None, None
    if args.mode == 'TRAIN':
        ff_acces, bp_acces = list(), list()

        print('\n\n')
        print(f'TRAIN MODEL'.center(60, '='))
        for _ in range(args.epoch):
            train_dataLoader = utils.mnistDataLoader(train=True, batch_size=args.train_batch_size)
            train(ff_model, bp_model, train_dataLoader, device=DEVICE)
            test_dataLoader = utils.mnistDataLoader(train=False, batch_size=args.test_batch_size)
            ff_acc, bp_acc = inference(ff_model, bp_model, test_dataLoader, device=DEVICE)
            ff_acces.append(ff_acc)
            bp_acces.append(bp_acc)
            print('')
        print('=' * 60)
        
        #save figure of accuracy
        utils.save_plot(  
            ff_acces, bp_acces, figure_path=os.path.join(FIGURE_HOME, 'figure1_accuracy_on_testset.png')
        )

        #save model
        print('\n\n')
        print('Save Model'.center(60, '='))
        torch.save(ff_model.state_dict(), FF_PATH)
        torch.save(bp_model.state_dict(), BP_PATH)

        print(f'\tFF Model: {FF_PATH}')
        print(f'\tBP Model: {BP_PATH}')
        print('=' * 60)

    elif args.mode == 'INFERENCE':
        print('\n\n')
        print(f'INFERENCE'.center(60, '='))
        test_dataLoader = utils.mnistDataLoader(train=False, batch_size=args.test_batch_size)
        ff_acc, bp_acc = inference(ff_model, bp_model, test_dataLoader, device=DEVICE)
        print('=' * 60)


    #Common code of Train & Inference
    print('\n\n')
    print(f'INFERENCE RESULT'.center(60, '='))
    print(f'+ Accuracy on MNIST Test Set')
    print(f'\tFF Model: {ff_acc:.3f}')
    print(f'\tBP Model: {bp_acc:.3f}')
    print('=' * 60)