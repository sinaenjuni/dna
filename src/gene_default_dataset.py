import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class GeneRNADataset(Dataset):
    def __init__(self, fn):
        data = pd.read_csv(fn)
        data = np.array(data)

        self.miRNA_vec = torch.from_numpy(data[:, :128].astype(np.float32))
        self.Gene_vec = torch.from_numpy(data[:, 128:256].astype(np.float32))
        self.label = torch.from_numpy(data[:, 258:259].astype(np.float32))

        self.miRNA_name = data[:, 256:257].astype('str')
        self.Gene_name = data[:, 256:257].astype('str')


    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return [self.miRNA_vec[index],
                self.Gene_vec[index],
                self.label[index]]

class GeneRNADataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32, train_ratio=0.8):
        super(GeneRNADataModule, self).__init__()

        self.batch_size = batch_size
        self.gn_dataset = GeneRNADataset('~/data/dna/AXB_383_gene_default.csv')

        N = len(self.gn_dataset)
        tr = int(N * train_ratio)  # 8 for the training
        va = N - tr  # 2 for the validation
        self.train_dataset, self.valid_dataset = random_split(self.gn_dataset, [tr, va])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    gn_dataset = GeneRNADataset('~/data/dna/AXB_383_gene_default.csv')
    gn_dataset.label.unique(return_counts=True)


    gn_data_module = GeneRNADataModule(batch_size=32, train_ratio=0.8)
    print(gn_data_module)
    print(gn_data_module.batch_size)
    miRNA_vec, Gene_vec, label = iter(gn_data_module.train_dataloader()).__next__()
    print(miRNA_vec.shape)
    print(Gene_vec.shape)
    print(label.shape)
    print(label)
    #
    # data = pd.read_csv('~/data/dna/AXB_383_gene_default.csv')
    # data = np.array(data)
    #
    # miRNA_vec_float = torch.from_numpy(data[:, :128].astype(np.float64))
    # Gene_vec = data[:, 128:256]
    # miRNA_name = data[:, 256:257].astype('str')
    # Gene_name = data[:, 256:257].astype('str')
    # label = data[:, 258:259].astype(np.int_)
    #
    #
    #
    # gn_dataset = GeneRNADataset('~/data/dna/AXB_383_gene_default.csv')
    #
    # N = len(gn_dataset)
    # tr = int(N * 0.8)  # 8 for the training
    # va = N - tr  # 2 for the validation
    # train_dataset, valid_dataset = random_split(gn_dataset, [tr, va])
    #
    #
    # miRAN_vec, Gene_vec, label = train_dataset[:10]
    #
    # print(miRAN_vec.shape)
    # print(Gene_vec.shape)
    # print(label.shape)