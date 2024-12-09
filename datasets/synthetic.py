import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

# TODO: Need to do a some cleaning here
#       Main problems are:
#           * we generally assume treatments and outcomes are scalars, eq 11 doesn't make sense then
#           * need to add support for noisy tests

class SyntheticDataset(Dataset):
    def __init__(self, z_dim: int = 5, 
                 mu_1: float = 5, 
                 mu_0: float = 3, 
                 artificial_size: int = 1000,
                 correlated: bool = False):
        """
        NOTE: synthetic set-up similar to https://arxiv.org/pdf/1705.08821, assume equal number
            of observeables, latents, outcomes and treatments
        """

        # features 0-5 are continuous, 6-24 are binary
        #print("Artificial size-", artificial_size)
        self.num_categories = {i : np.inf for i in range(z_dim)}

        self.z_dim = z_dim
        self.mu_1 = mu_1
        self.mu_0 = mu_0

        self.correlated = correlated
        
        # NOTE: E[E[X]] = E[X]
        #       Mean regardless of correlation is 0.5
        # self.y_p = 1 / (1 + np.exp(-1.5))
        # self.y_mean, self.y_std = self.y_p, np.sqrt(self.y_p * (1 - self.y_p))

        self.y_mean, self.y_std = 0, 1

        self.artifical_size = artificial_size

    def __len__(self):
        return self.artifical_size

    def __getitem__(self, idx):
        if self.correlated:
            z = torch.distributions.MultivariateNormal(
                torch.ones((self.z_dim)) / 2, 
                scale_tril=torch.tril(torch.ones(self.z_dim, self.z_dim)))
            z = z.sample()
            z = torch.sigmoid(z)
        else:
            z = torch.bernoulli(0.5 * torch.ones(self.z_dim)).to(torch.double)
            
        x = torch.normal(z, (self.mu_1 ** 2) * z + (self.mu_0 ** 2) * (1 - z)).to(torch.double)

        # TODO: we expect scalar treatment and outcomes!
        #       this doesn't seem to line up well with eq 11: https://arxiv.org/pdf/1705.08821
        t = torch.bernoulli(0.75 * z + 0.25 * (1 - z)).to(torch.double)
        t = t[0]
        
        y = torch.bernoulli(torch.sigmoid(3 * (z + 2 * (2 * t - 1)))).to(torch.double)
        y = y[0]
        # counterfactual by reversing value of t
        y_cf = torch.bernoulli(torch.sigmoid(3 * (z + 2 * (2 * (1 - t) - 1)))).to(torch.double)
        y_cf = y_cf[0]

        return np.array(self.mu_1, dtype=np.double), np.array(self.mu_0, dtype=np.double),\
            np.array(t, dtype=np.double), np.array(x, dtype=np.double),\
            np.array(y, dtype=np.double), np.array(y_cf, dtype=np.double), \
            np.array((y - self.y_mean) / self.y_std, dtype=np.double)

    def indices_each_features_misspecified(self):
        return \
            [(x, 2) for x in range(self.z_dim)],\
            []
    
    def indices_each_features(self):
        return \
            [],\
            [(x, np.inf) for x in range(self.z_dim)]
    
def get_SyntheticDataloader(batch_size:int, curr_dataset = None, val_fraction: float = 0.3, test_fraction: float = 0.1):
    """
    NOTE: Default splits are 63% : 27% : 10% for training : validation : testing, repsectively
    """

    if curr_dataset == None:
        curr_dataset = SyntheticDataset()

    split_fractions = [(1 - val_fraction) * (1 - test_fraction), (val_fraction) * (1 - test_fraction), test_fraction]
    train_split, val_split, test_split = torch.utils.data.random_split(curr_dataset, split_fractions)

    train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_split, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=batch_size)

    return train_loader, val_loader, test_loader