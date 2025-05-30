from torch.utils.data import Dataset
from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors

MAX_X=30
MAX_Y=30
NUM_VALUES=10

def one_hot_tensor(in_grid):
    """Transform the grid into a (max_x, max_y, num_values) sized one hot tensor"""
    in_tensor = torch.tensor(in_grid, dtype=torch.long)+1
    in_x, in_y = in_tensor.shape
    padded_tensor = torch.zeros((MAX_X, MAX_Y), dtype=torch.long)
    padded_tensor[0:in_x, 0:in_y] = in_tensor
    one_hot = F.one_hot(padded_tensor, NUM_VALUES+1).permute((2,0,1))
    return one_hot.to(torch.float32)

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
        train_inputs = []
        train_outputs = []
        for example in challenge['train']:
            train_inputs.append(one_hot_tensor(example['input']))
            train_outputs.append(one_hot_tensor(example['output']))
        test_inputs = []
        test_outputs = []
        for example in challenge['test']:
            test_inputs.append(one_hot_tensor(example['input']))
        for example in solution:
            test_outputs.append(one_hot_tensor(example))

        return (torch.stack(train_inputs), torch.stack(train_outputs),
                torch.stack(test_inputs), torch.stack(test_outputs))

MAX_EXAMPLES=10
@dataclass
class SequenceLocation:
    data_idx: str
    test_idx: int

class ArcSequence(Dataset):

    def __init__(self, data_folder, prefix):
        super().__init__()
        with open(f'{data_folder}/{prefix}_challenges.json') as f:
            self.challenges = json.load(f)
        with open(f'{data_folder}/{prefix}_solutions.json') as f:
            self.solutions = json.load(f)
        problems = list(self.challenges.keys())
        self.sequence_locations = []
        for problem in problems:
            for test in range(len(self.challenges[problem]['test'])):
                self.sequence_locations.append(SequenceLocation(problem, test))

    def __len__(self):
        return len(self.sequence_locations)

    def __getitem__(self, idx):
        location = self.sequence_locations[idx]
        challenge = self.challenges[location.data_idx]
        solution = self.solutions[location.data_idx]
        x = torch.zeros(2*MAX_EXAMPLES+1, NUM_VALUES+1, MAX_X, MAX_Y)
        for i, example in enumerate(challenge['train']):
            x[2*i,:,:,:] = one_hot_tensor(example['input'])
            x[2*i+1,:,:,:] = one_hot_tensor(example['output'])
        x[-1,:,:,:] = one_hot_tensor(challenge['test'][location.test_idx]['input'])
        y = one_hot_tensor(solution[location.test_idx])
        return x,y

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
        np.random.shuffle(negative)
        negative = negative.reshape(positive.shape)
        return (one_hot_tensor(anchor),
                one_hot_tensor(positive),
                one_hot_tensor(negative))

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

CMAP = colors.ListedColormap(['#FFFFFF', '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

def print_grid(grid, axis, cmap=CMAP):
    val_grid = grid.argmax(dim=0)
    has_values = val_grid > 0
    height = int(has_values.sum(dim=0).max().item())
    width = int(has_values.sum(dim=1).max().item())
    mat = val_grid[:height,:width].tolist()
    axis.imshow(mat, cmap=cmap, vmin=0, vmax=NUM_VALUES)

def print_problem(train_inputs, train_outputs, test_inputs, test_outputs, show_solution=False):
    n_train = train_inputs.shape[0]
    n_test = test_inputs.shape[0]
    fig, axs = plt.subplots(n_train+n_test, 2)
    for i in range(n_train):
        print_grid(train_inputs[i,:,:], axs[i,0])
        axs[i,0].set_axis_off()
        print_grid(train_outputs[i,:,:], axs[i,1])
        axs[i,1].set_axis_off()
    for i in range(n_test):
        print_grid(test_inputs[i,:,:], axs[n_train+i,0])
        axs[n_train+i,0].set_axis_off()
        if show_solution:
            print_grid(test_outputs[i,:,:], axs[n_train+i,1])
        axs[n_train+i,1].set_axis_off()
    return fig
