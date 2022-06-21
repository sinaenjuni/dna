from typing import Union

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model_wandb import MyModel
from embeddings_reproduction import embedding_tools

app = FastAPI()

data_rna_protein = pd.read_csv('~/data/dna/protein/protein2vec_rna.csv')
data_gene_protein = pd.read_csv('~/data/dna/protein/protein2vec_gene.csv')
data_core_protein = pd.read_csv('~/data/dna/protein/protein2vec_default.csv')

def getRnaData(name = ['hsa-miR-98-5p', 'hsa-let-7a-5p']):
    seq = data_rna_protein[data_rna_protein['miRNA_name'].isin(name)].filter(regex='miRNA_seq').values
    vec = data_rna_protein[data_rna_protein['miRNA_name'].isin(name)].filter(regex='rna_pro2vac_.').values
    return seq, vec

def getGeneData(name = ['hsa-miR-98-5p', 'hsa-let-7a-5p']):
    seq = data_gene_protein[data_gene_protein['gene_name'].isin(name)].filter(regex='gene_seq').values
    vec = data_gene_protein[data_gene_protein['gene_name'].isin(name)].filter(regex='gene_pro2vac_.').values
    return seq, vec


class Name(BaseModel):
    name: str

class PredictDF(BaseModel):
    vec: list


@app.get("/data/{page}")
def getDatas(page : int):
    # return data_core_protein[["0_mirna","1_gene","label"]]
    return {"rna":list(data_core_protein["0_mirna"][page:page+5]),
            "gene":list(data_core_protein["1_gene"][page:page+5]),
            "label":list(data_core_protein["label"][page:page+5])
            }


@app.post("/rna")
def getRna(name : Name):
    print(name)
    key = name.name
    try:
        seq = data_rna_protein[data_rna_protein['miRNA_name']==key].filter(regex='miRNA_seq').values[0]
        vec = data_rna_protein[data_rna_protein['miRNA_name']==key].filter(regex='rna_pro2vac_.').values[0]
    except IndexError:
        seq = "None"
        vec = "None"
    return {"name": key,
            "seq": str(seq),
            "vec": list(vec)}

@app.post("/gene")
def getGene(name : Name):
    key = name.name
    try:
        seq = data_gene_protein[data_gene_protein['gene_name']==key].filter(regex='gene_seq').values[0]
        vec = data_gene_protein[data_gene_protein['gene_name']==key].filter(regex='gene_pro2vac_.').values[0]
    except IndexError:
        seq = "None"
        vec = "None"

    return {"name": key,
            "seq": str(seq),
            "vec": list(vec)}

@app.post("/pred")
def getSeqVec(rna: PredictDF, gene : PredictDF):
    rna_vec = np.array(rna.vec).astype(np.float32)
    gene_vec = np.array(gene.vec).astype(np.float32)

    print(rna_vec)
    print(gene_vec)

    # rna_name = rna.name
    # rna_seq = rna
    #
    # gene_name = gene.name
    #
    #
    # rna_vec =
    #     gene_seq, gene_vec = getGeneData(gene_name)
    #
    # print(rna_vec.shape)
    # print(gene_vec.shape)

    # print(rna_seq, rna_vec)
    # print(gene_seq, gene_vec)

    model = MyModel.load_from_checkpoint('model_weight.ckpt')
    model.eval()
    logits, _ = model(torch.from_numpy(rna_vec), torch.from_numpy(gene_vec))
    print(logits)
    return {"logit": str(logits)}
    #
    # print(logits)

    # print(type(rna_emb))
    # print(type(gene_emb))
    # return {"rna_seq": str(rna_seq),
    #         "rna_vec": str(rna_vec),
    #         "gene_seq": str(gene_seq),
    #         "gene_vec": str(gene_vec),}
    #         "pred": str(logits)}
    # return rna_seq, rna_vec

