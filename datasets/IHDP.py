import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class IHDPDataset(Dataset):
    def __init__(self, data_path: str = "./datasets/IHDP", replication: int = 0):
        """
        replication: indicates which replication of IHDP to read from (if you'd like to do a different replication change this variable)
            values 1-10 are supported and all appear to have the same length
        """

        # features 0-5 are continuous, 6-24 are binary
        self.num_categories = {i : np.inf if i <= 5 else 2 for i in range(25)}

        # inputs: (x, t, y), labels: (y_cf, mu_0, mu_1)
        curr_data = np.loadtxt(f"{data_path}/ihdp_npci_{str(replication + 1)}.csv", delimiter=",")
        self.t = curr_data[:, 0]
        self.y = curr_data[:, 1]
        self.y_cf = curr_data[:, 2]
        self.mu_0 = curr_data[:, 3]
        self.mu_1 = curr_data[:, 4]
        self.x = curr_data[:, 5:]

        self.x[:, 13] -= 1  # {1, 2} -> {0, 1}

        self.y_mean, self.y_std = np.mean(self.y), np.std(self.y)
        # self.y_mean, self.y_std = 0, 1
        self.standard_y = (self.y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.mu_1[idx], self.mu_0[idx], self.t[idx], self.x[idx], self.y[idx], self.y_cf[idx], self.standard_y[idx]
    
    def indices_each_features(self):
        return \
            [(x, 2) for x in range(6, 25)],\
            [(x, np.inf) for x in range(6)]
    
def get_IHDPDataloader(batch_size: int, curr_dataset = None, replication: int = 0, val_fraction: float = 0.3, test_fraction: float = 0.1):
    """
    NOTE: Default splits are 63% : 27% : 10% for training : validation : testing, repsectively
    """

    if curr_dataset == None:
        curr_dataset = IHDPDataset(replication=replication)

    split_fractions = [(1 - val_fraction) * (1 - test_fraction), (val_fraction) * (1 - test_fraction), test_fraction]
    train_split, val_split, test_split = torch.utils.data.random_split(curr_dataset, split_fractions)

    train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_split, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=batch_size)

    return train_loader, val_loader, test_loader