import os
import pandas as pd
import numpy as np
pd.describe_option('display')
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

PATH = '~/data/dna/'
RNA_DATA_PATH = os.path.join(PATH, "mirna_seq_22v_We.csv")
GENE_DATA_PATH = os.path.join(PATH,"NEdit_mirBase_gene_distinct_list.csv")
RELATION_DATA_PATH = os.path.join(PATH, "hsa_MTI.xlsx")
DATA_PATH = os.path.join(PATH, "AXB_383_gene_default.csv")
rna_data = pd.read_csv(RNA_DATA_PATH)
gene_data = pd.read_csv(GENE_DATA_PATH)
relation_data = pd.read_excel(RELATION_DATA_PATH)
data = pd.read_csv(DATA_PATH)


rna_seq, rna_name = rna_data.keys()
gen_name, gen_seq = gene_data.keys()

print(len(rna_data), len(gene_data))
print(len(rna_data[rna_seq].min()), len(rna_data[rna_seq].max()))
print(len(gene_data[gen_seq].min()), len(gene_data[gen_seq].max()))


result = []
for i in gene_data[gen_seq].to_numpy():
    for c in i:
        result.append(c)




rna_data[name].value_counts()[rna_data[name].value_counts()>0]
overlap_seqs = rna_data[seq].value_counts()[rna_data[seq].value_counts()>1].keys().to_list()

for overlap_seq in overlap_seqs:
    print(rna_data[rna_data[seq] == overlap_seq])


name, seq = gene_data.keys()
gene_data[name].value_counts()[gene_data[name].value_counts()>1]
overlap_seqs = gene_data[seq].value_counts()[gene_data[seq].value_counts()>1].keys().to_list()

for overlap_seq in overlap_seqs:
    print(gene_data[gene_data[seq] == overlap_seq])