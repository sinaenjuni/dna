import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

data = pd.read_csv('~/data/dna/AXB_383_gene_default.csv')
data = np.array(data)


class GeneRNADataset(Dataset):
    def __init__(self, fn):
        data = pd.read_csv(fn)
        data = np.array(data)

        self.miRNA_vec = data[:, :128]
        self.Gene_vec = data[:, 128:256]
        self.miRNA_name = data[:, 256:257]
        self.Gene_name = data[:, 256:257]
        self.label = data[:, 258:259]

    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return [self.miRNA_vec,
                self.Gene_vec,
                self.label]

class GeneRNADataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32, train_ratio=0.8):
        self.batch_size = batch_size
        self.gn_dataset = GeneRNADataset('~/data/dna/AXB_383_gene_default.csv')

        N = len(self.gn_dataset)
        tr = int(N * train_ratio)  # 8 for the training
        va = N - tr  # 2 for the validation
        self.train_dataset, self.valid_dataset = random_split(self.gn_dataset, [tr, va])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)


gn_dataset = GeneRNADataset('~/data/dna/AXB_383_gene_default.csv')

N = len(gn_dataset)
tr = int(N * 0.8)  # 8 for the training
va = N - tr  # 2 for the validation
train_dataset, valid_dataset = random_split(gn_dataset, [tr, va])


miRAN_vec, Gene_vec, label = gn_dataset[1]

print(miRAN_vec.shape)
print(Gene_vec.shape)
print(label.shape)