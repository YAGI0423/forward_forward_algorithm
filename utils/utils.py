import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from matplotlib import pyplot as plt


def mnistDataLoader(train: bool, batch_size: int) -> DataLoader:
    '''
    pytorch MNIST train & test 데이터 로더 반환
    + z-normalization
    + Flatten
    '''
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.1307, ), std=(0.3081, )),
        Lambda(lambda x: torch.flatten(x)),
    ])

    loader = DataLoader(
        MNIST(
            root='./mnist/',
            train=train,
            transform=transform,
            download=True
        ), shuffle=train, batch_size=batch_size
    )
    return loader

def save_plot(ff_acc: list, bp_acc: list, figure_path: str) -> None:
    plt.figure(figsize=(14, 5.5))
    plt.suptitle('Accuracy on Test Set', fontsize=15, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.125, top=0.825, wspace=0.175, hspace=0.1)
    
    plt.subplot(1, 2, 1)
    plt.title('ff', fontdict={'fontsize': 13, 'fontweight': 'bold'}, loc='left', pad=10)
    plt.plot(ff_acc, color='green')
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.title('bp', fontdict={'fontsize': 13, 'fontweight': 'bold'}, loc='left', pad=10)
    plt.plot(bp_acc, color='black')
    plt.grid()
    plt.savefig(figure_path)