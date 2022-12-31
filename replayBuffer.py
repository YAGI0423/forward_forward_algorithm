import torch
import numpy as np

class Buffer:
    def __init__(self, buffer_size: int, batch_size: int=1, device='cpu'):
        self.__buf = {'state': [], 'action': [], 'reward': [], 'state_n': [],}
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

    def add_buffer(self, **kwargs):
        for key, value in kwargs.items():
            v = value.tolist()
            self.__buf[key].append(v)

    def update(self):
        for key, value in self.__buf.items():
            self.__buf[key] = value[-self.buffer_size:]

    def get_batch(self):
        buffer_size = len(self.__buf['state'])
        batch_idx = np.random.randint(buffer_size, size=self.batch_size)
        return tuple(self.__convertToTensor(np.array(value)[batch_idx]) for value in self.__buf.values())

    def __convertToTensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)
