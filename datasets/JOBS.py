import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

RANDOMIZED_TRIAL_SIZE = 722

class JOBSDataset(Dataset):
    def __init__(self, data_path: str = "./datasets/JOBS"):
        """
        NOTE: This dataset does not have counterfactuals, but by virtue of being a
            randomized study allows for easy estimation of the ATT (difference of average RE78)

        The order of the variables from left to right is: 

            treatment indicator (1 if treated, 0 if not treated), 
            age, 
            education, 
            Black (1 if black, 0 otherwise), 
            Hispanic (1 if Hispanic, 0 otherwise), 
            married (1 if married, 0 otherwise), 
            nodegree (1 if no degree, 0 otherwise), 
            RE75 (earnings in 1975), and 
            RE78 (earnings in 1978). 

        The last variable is the outcome; other variables are pre-treatment. 

        NOTE: We overlord the y_cf field to use for marking whether or not a
              datapoint comes from the LaLonde randomized trial or not (PSID comparison).
              THAT IS TO SAY, DO NOT USE THE y_cf FIELD UNLESS YOU KNOW WHAT YOU'RE DOING!
        """

        # features 1, 2, 7, 8 are continuous, 0, 3-6 are binary
        # move 7, 8 to 2, 1, push 1-6 to 3-8, then re-index after cutting off 0, 1
        self.num_categories = {
            0: np.inf,
            1: np.inf,
            2: np.inf,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
        }

        # inputs: (x, t, y), labels: (y_cf, mu_0, mu_1)
        curr_data = np.loadtxt(f"{data_path}/nsw_merged.txt")
        self.t = curr_data[:, 0]
        self.y = (curr_data[:, 8] > 0).astype(np.float64)
        # NOTE: This is a hack to overload the otherwise not used
        self.y_cf = (np.arange(curr_data.shape[0]) > RANDOMIZED_TRIAL_SIZE).astype(np.float64)
        self.mu_0 = None
        self.mu_1 = None
        self.x = np.concatenate([curr_data[:, 1:3], curr_data[:, 7][:, np.newaxis], curr_data[:, 3:7]], axis=1)

        # self.y_mean, self.y_std = np.mean(self.y), np.std(self.y)
        self.y_mean, self.y_std = 0, 1
        self.standard_y = (self.y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return -1, -1, self.t[idx], self.x[idx].squeeze(), self.y[idx], self.y_cf[idx].squeeze(), self.standard_y[idx]
    
    def indices_each_features(self):
        return \
            [(x, categories) for x, categories in self.num_categories.items() if categories != np.inf],\
            [(x, np.inf) for x, categories in self.num_categories.items() if categories == np.inf]

# NOTE: Slightly different train/val/test split sizes (https://arxiv.org/pdf/1606.03976)
def get_JOBSDataloader(batch_size:int, curr_dataset = None, val_fraction: float = 0.24, test_fraction: float = 0.2):
    """
    NOTE: Default splits are 56% : 24% : 20% for training : validation : testing, repsectively (matching IHDP?)
    """

    if curr_dataset == None:
        curr_dataset = JOBSDataset()

    split_fractions = [(1 - val_fraction) * (1 - test_fraction), (val_fraction) * (1 - test_fraction), test_fraction]
    train_split, val_split, test_split = torch.utils.data.random_split(curr_dataset, split_fractions)

    train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_split, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=batch_size)

    return train_loader, val_loader, test_loader