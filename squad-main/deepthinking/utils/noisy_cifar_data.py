
import torch

from torch.utils import data
from easy_to_hard_data import ChessPuzzleDataset

import os

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


# class FlippedChessPuzzleDataset(ChessPuzzleDataset):
#     """Class to get flipped chess data. In this setting the player to move next is always
#     at the bottom of the board, and the king is always on the right"""
#     def __init__(self, root: str,
#                  train: bool = True,
#                  idx_start: int = None,
#                  idx_end: int = None,
#                  who_moves: bool = True,
#                  download: bool = True):
#         ChessPuzzleDataset.__init__(self, root, train, idx_start, idx_end, who_moves, download)
#         rotate_idx = (self.who_moves == 1).squeeze()
#         rotated_puzzles = torch.flip(self.puzzles[rotate_idx], [2])
#         self.puzzles[rotate_idx] = rotated_puzzles
#         rotated_targets = torch.flip(self.targets[rotate_idx], [1])
#         self.targets[rotate_idx] = rotated_targets


class NoisyImageDataset(torch.utils.data.Dataset):
    base_folder = "noisy_image_data"
    url = "https://www.dropbox.com/s/gamc8j5vqbvushj/noisy_image_data.tar.gz"
    lengths = [0.1,0.2,0.3,0.4,0.5]
    download_list = [f"data_{l}.pth" for l in lengths] + [f"targets_{l}.pth" for l in lengths]

    def __init__(self, root: str, num_bits: float = 0.1, download: bool = True, train: bool = True):

        self.root = root

#         if download:
#             self.download()

        print(f"Loading data with {num_bits} bits.")

        # TODO: swap inputs and targets path
        targets_path = os.path.join(root, self.base_folder, f"data_{num_bits}.pth")
        inputs_path = os.path.join(root, self.base_folder, f"targets_{num_bits}.pth")
        self.inputs = torch.tensor(torch.load(inputs_path)).float()
        self.targets = torch.load(targets_path)
        self.train = train

        if train:
            print("Training data using pre-set indices [0, 8000).")
            idx_start = 0
            idx_end = 8000
        else:
            print("Testing data using pre-set indices [8000, 10000).")
            idx_start = 8000
            idx_end = 10000

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.download_list:
            fpath = os.path.join(root, self.base_folder, fentry)
            if not os.path.exists(fpath):
                return False
        return True

    # def download(self) -> None:
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     path = download_url(self.url, self.root)
    #     extract_zip(path, self.root)
    #     os.unlink(path)
        

# def get_dataloaders(train_batch_size, test_batch_size, shuffle=True):

#     train_data = NoisyImageDataset("./data", train=True)
#     test_data = NoisyImageDataset("./data", train=False,num_bits=0.5)
#     # train_data = MazeDataset("./data", train=True)
#     # test_data = MazeDataset("./data", train=False)



#     trainloader = data.DataLoader(train_data, num_workers=0, batch_size=train_batch_size,
#                                   shuffle=shuffle, drop_last=True)
#     testloader = data.DataLoader(test_data, num_workers=0, batch_size=test_batch_size,
#                                  shuffle=False, drop_last=False)
#     return trainloader, testloader



def prepare_image_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    # trainset = FlippedChessPuzzleDataset("./data", idx_start=0, idx_end=train_data, who_moves=False,
    #                                      download=True)
    # testset = FlippedChessPuzzleDataset("./data", idx_start=test_data-100000, idx_end=test_data,
    #                                     who_moves=False, download=True)
    
    root = '/fs/clip-quiz/amao/Github/easy-to-hard/denoise/data/'
    trainset = NoisyImageDataset(root, train=True)
    testset = NoisyImageDataset(root, train=False,num_bits=0.5)


    train_split = int(0.8 * len(trainset))

    trainset, valset = torch.utils.data.random_split(trainset,
                                                     [train_split, int(len(trainset)-train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset, num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset, num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset, num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
