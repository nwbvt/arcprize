from torch.utils.data import Dataset
import json

class ArcData(Dataset):

    def __init__(self, data_folder, prefix, max_x=30, max_y=30):
        self.max_x = max_x
        self.max_y = max_y
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
        
