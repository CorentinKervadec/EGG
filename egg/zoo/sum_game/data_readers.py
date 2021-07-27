# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils.data import Dataset

# These input-data-processing classes take input data from a text file and convert them to the format
# appropriate for the recognition and discrimination games, so that they can be read by
# the standard pytorch DataLoader. The latter requires the data reading classes to support
# a __len__(self) method, returning the size of the dataset, and a __getitem__(self, idx)
# method, returning the idx-th item in the dataset. We also provide a get_n_features(self) method,
# returning the dimensionality of the Sender input vector after it is transformed to one-hot format.

# The SumDataset class is constructed from the AttValRecoDataset class is used in the reconstruction game. It takes an input file with a
# space-delimited attribute-value vector per line and  creates a data-frame with the two mandatory
# fields expected in EGG games, namely sender_input and labels.
# In this case, the two fields contain the same information, namely the input attribute-value vectors,
# represented as one-hot in sender_input, and in the original integer-based format in
# labels.
class SumDataset(Dataset):
    def __init__(self, path, n_range):
        frame = open(path, "r")
        self.frame = []
        for row in frame:
            raw_info = row.split(".")# 'x y . x+y'
            # target = x+y in [0, 2N]
            label = int(raw_info[-1])
            # label = np.zeros(n_range*2)
            # label[target_sum]=1
            # input = (x, y) in [0, N]
            input_int = list(map(int, raw_info[0].split()))
            z = torch.zeros((2, n_range))
            for i in range(2):
                z[i, input_int[i]] = 1
            self.frame.append((z.view(-1), label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

class BinarySumDataset(Dataset):
    def __init__(self, path, n_range):
        frame = open(path, "r")
        self.frame = []
        for row in frame:
            raw_info = row.split(".")# 'x y . x+y'
            # target = x+y in [0, 2N]
            label = int(raw_info[-1])
            # label = np.zeros(n_range*2)
            # label[target_sum]=1
            # input = (x, y) in [0, N]
            input_int = list(map(int, raw_info[0].split()))
            
            m = int(np.log2(n_range)+1)
            z = torch.zeros((2, m))
            for i in range(2):
                z[i] = torch.from_numpy(self.bin_array(input_int[i], m))
            self.frame.append((z.view(-1), label))

    def bin_array(self, num, m):
        """Convert a positive integer num into an m-bit bit vector"""
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]


# The AttValDiscriDataset class, used in the discrimination game takes an input file with a variable
# number of period-delimited fields, where all fields but the last represent attribute-value vectors
# (with space-delimited attributes). The last field contains the index (counting from 0) of the target
# vector.
# Here, we create a data-frame containing 3 fields: sender_input, labels and receiver_input (these are
# expected by EGG, the first two mandatorily so).
# The sender_input corresponds to the target vector (in one-hot format), labels are the indices of the
# target vector location and receiver_input is a matrix with a row for each input vector (in input order).
class AttValDiscriDataset(Dataset):
    def __init__(self, path, n_values):
        frame = open(path, "r")
        self.frame = []
        for row in frame:
            raw_info = row.split(".")
            index_vectors = list([list(map(int, x.split())) for x in raw_info[:-1]])
            target_index = int(raw_info[-1])
            target_one_hot = []
            for index in index_vectors[target_index]:
                current = np.zeros(n_values)
                current[index] = 1
                target_one_hot = np.concatenate((target_one_hot, current))
            target_one_hot_tensor = torch.FloatTensor(target_one_hot)
            one_hot = []
            for index_vector in index_vectors:
                for index in index_vector:
                    current = np.zeros(n_values)
                    current[index] = 1
                    one_hot = np.concatenate((one_hot, current))
            one_hot_sequence = torch.FloatTensor(one_hot).view(len(index_vectors), -1)
            label = torch.tensor(target_index)
            self.frame.append((target_one_hot_tensor, label, one_hot_sequence))
        frame.close()

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]
