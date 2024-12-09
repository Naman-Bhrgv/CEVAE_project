import json

import numpy as np
import pandas
import torch

from torch.utils.data import Dataset, DataLoader

# TODO: Need to do a lot more cleaning here
#       Main problems are:
#           * piping around categorical variables with different numbers of categories
#           * dealing with useless/bad covariates (infant_id_{0,1}, bord_{0,1}), currently popping
#           * figuring out exactly how this data was used (somewhat unclear)

# NOTE: We assume that T and Y are the variables as created in the paper?

class TWINSDataset(Dataset):
    def __init__(self, data_path: str = "./datasets/TWINS"):
        """
        NOTE: Drops rows with missing values

        NOTE: This dataset has counterfactuals by virtue of consisting of twins
              Only choosing twins of the same sex and born weighting less than 2kg
        """

        # read from files to figure out variables
        self.num_categories = {}

        with open(f"{data_path}/covar_type.txt", "r") as variable_map:
            variable_map = variable_map.read().strip().replace("\'", "\"")
            variable_map = json.loads(variable_map)

        # inputs: (x, t, y), labels: (y_cf, mu_0, mu_1)
        raw_data = pandas.read_csv(f"{data_path}/twin_pairs_X_3years_samesex.csv").dropna()

        for key, value in variable_map.items():
            if value == "bin":
                self.num_categories[key] = 2
            elif value == "cyc":
                # special case for months, see birmon?
                self.num_categories[key] = 12
            elif value == "ord":
                self.num_categories[key] = np.inf
            elif value == "cat":
                self.num_categories[key] = len(raw_data[key].unique())

        # bad covariates to remove (infant indices won't exist?)
        # NOTE: bord_0 and bord_1 are unified as bord
        self.num_categories.pop("bord")

        curr_treatment_raw = pandas.read_csv(f"{data_path}/twin_pairs_T_3years_samesex.csv").loc[raw_data.index].to_numpy()
        curr_outcome_raw = pandas.read_csv(f"{data_path}/twin_pairs_Y_3years_samesex.csv").loc[raw_data.index].to_numpy()

        # no need to clean off indexing column

        # need to push continuous variables to front of list (treatment, outcome, counterfactual_outcome, )
        self.sorted_keys = sorted(self.num_categories.keys(), key=lambda x : -1 * self.num_categories.get(x))

        # reorder columns, removes index by default
        curr_data = raw_data[self.sorted_keys].to_numpy()

        # treatment is whether or not they were the heavier twin (0 if lighter, 1 if heavier)
        # NOTE: randomly pick to stimulate random trial (per paper)
        self.random_trial = np.random.randint(0, 2, (len(curr_data)))

        self.t = np.array(
            [self.random_trial[idx] if curr_treatment_raw[idx, 0] < curr_treatment_raw[idx, 1]\
            else 1 - self.random_trial[idx]\
            for idx in range(curr_treatment_raw.shape[0])]
        ).astype(np.float64)

        # outcome is mortality
        self.y = np.array(
            [curr_outcome_raw[idx, self.random_trial[idx] + 1] for idx in range(len(curr_outcome_raw))]
        ).astype(np.float64)

        self.y_cf = np.array(
            [curr_outcome_raw[idx, 2 - self.random_trial[idx]] for idx in range(len(curr_outcome_raw))]
        ).astype(np.float64)

        # label is all 45 other features
        self.x = curr_data.astype(np.float64)

        self.y_mean, self.y_std = np.mean(self.y), np.std(self.y)
        # self.y_mean, self.y_std = 0, 1
        self.standard_y = (self.y - self.y_mean) / self.y_std

        # just in case
        self.t = self.t.squeeze()
        self.y = self.y.squeeze()
        self.x = self.x.squeeze()
        self.standard_y = self.standard_y.squeeze()

        # HACK: Overloading the conditional mean outcomes for now (this is a randomized trial)
        self.mu1 = np.sum(self.t * self.y) / np.sum(self.t)
        self.mu0 = np.sum((1 - self.t) * self.y) / np.sum(1 - self.t)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.mu1, self.mu0, self.t[idx], self.x[idx], self.y[idx], self.y_cf[idx], self.standard_y[idx]
    
    # HACK: No guidance on how to deal with many categorical variables of different number of
    #       of categories, so for now we assume categorical variables are modeled via a
    #       normal distribution
    def indices_each_features(self):
        return \
            [(x, 2) for x in self.sorted_keys if self.num_categories[x] == 2],\
            [(x, self.num_categories[x]) for x in self.sorted_keys if self.num_categories[x] != 2]

def get_TWINSDataloader(batch_size:int, curr_dataset = None, val_fraction: float = 0.3, test_fraction: float = 0.1):
    """
    NOTE: Default splits are 63% : 27% : 10% for training : validation : testing, repsectively
    """

    if curr_dataset == None:
        curr_dataset = TWINSDataset()

    split_fractions = [(1 - val_fraction) * (1 - test_fraction), (val_fraction) * (1 - test_fraction), test_fraction]
    train_split, val_split, test_split = torch.utils.data.random_split(curr_dataset, split_fractions)

    train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_split, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=batch_size)

    return train_loader, val_loader, test_loader