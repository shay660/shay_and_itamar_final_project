from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            self.sequences, self.degradation_rates = args[0]
        elif len(args) == 2:
            self.sequences, self.degradation_rates = args
        else:
            raise ValueError(
                "Invalid arguments passed. Expected a tuple or two separate arguments.")

    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        return self.sequences[idx], self.degradation_rates[idx]
