import os, glob, torch
from torch.utils.data import Dataset, DataLoader

class PtDataset(Dataset):
    def __init__(self, pt_dir, n_files=None):
        self.files = sorted(
            glob.glob(os.path.join(pt_dir, '*.pt')),
            key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        )
        if n_files is not None:
            self.files = self.files[:n_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data