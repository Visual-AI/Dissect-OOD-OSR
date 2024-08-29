import numpy as np
import torch
from bisect import bisect_left


class TinyImages(torch.utils.data.Dataset):

    def __init__(self, data_path='/home/hjwang/osrd/data/300K_random_images.npy', transform=None):
        self.data = np.load(data_path)
        self.transform = transform
        self.offset = 0     # offset index

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        index = (index + self.offset) % len(self.data)

        img = self.data[index]
        # img = np.transpose(img, [2, 0, 1])

        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class