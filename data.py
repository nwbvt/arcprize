from torch.utils.data import Dataset
from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn.functional as F

MAX_X=30
MAX_Y=30
NUM_VALUES=10

def one_hot_tensor(in_grid):
    """Transform the grid into a (max_x, max_y, num_values) sized one hot tensor"""
    in_tensor = torch.tensor(in_grid, dtype=torch.long)
    in_x, in_y = in_tensor.shape
    padded_tensor = torch.zeros((MAX_X, MAX_Y), dtype=torch.long)
    padded_tensor[0:in_x, 0:in_y] = in_tensor
    return F.one_hot(padded_tensor, NUM_VALUES)

class ArcData(Dataset):

    def __init__(self, data_folder, prefix):
        with open(f'{data_folder}/{prefix}_challenges.json') as f:
            self.challenges = json.load(f)
        with open(f'{data_folder}/{prefix}_solutions.json') as f:
            self.solutions = json.load(f)
        self.problems = list(self.challenges.keys())

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, i):
        problem = self.problems[i]
        challenge = self.challenges[problem]
        solution = self.solutions[problem]
        return challenge, solution

@dataclass
class TripletLocation:
    data_idx: str
    anchor_idx: int
    comparison_idx: int
    side: str

    def get_pair(self, challenges):
        challenge = challenges[self.data_idx]['train']
        anchor = np.array(challenge[self.anchor_idx][self.side])
        positive = np.array(challenge[self.comparison_idx][self.side])
        return anchor, positive

    def gen_triplet(self, challenges):
        anchor, positive = self.get_pair(challenges)
        negative = positive.flatten()
        np.shuffle(negative_flat)
        negative = negative.reshape(positive.shape)
        return (one_hot_encode(anchor),
                one_hot_encode(positive),
                one_hot_encode(negativer))

    def big_enough(self, challenges, min_size):
        pair = self.get_pair(challenges)
        for item in pair:
            for dim in item.shape:
                if dim < min_size:
                    return False
        return True

class ArcTriplets(Dataset):

    def __init__(self, data_location, prefix, min_size):
        self.triplets = []
        with open(f'{data_location}/{prefix}_challenges.json') as f:
            self.challenges = json.load(f)
        for i in self.challenges:
            problem = self.challenges[i]['train']
            n = len(problem)
            for j in range(n):
                for k in range(n):
                    if j != k:
                        for side in ("input", "output"):
                            triplet = TripletLocation(i, j, k, side)
                            if triplet.big_enough(self.challenges, min_size):
                                self.triplets.append(triplet)
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx].gen_triplet(self.challenges)
