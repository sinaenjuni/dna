from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from embeddings_reproduction import embedding_tools

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

class Seqs(BaseModel):
    rna_seq: list
    gene_seq: list

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.post("/seq")
def mkSeq(seqs: Seqs):
    rna_seq = seqs.rna_seq
    gene_seq = seqs.gene_seq

    rna_emb = embedding_tools.get_embeddings_new('original_5_7.pkl', rna_seq, k=5, overlap=False)
    gene_emb = embedding_tools.get_embeddings_new('original_5_7.pkl', gene_seq, k=5, overlap=False)

    print(rna_emb)
    print(gene_emb)
    return {"rna_emb": str(rna_emb), "gene_emb": str(gene_emb)}

def getRelaive(seqs: Seqs):
    rna_seq = seqs.rna_seq
    gene_seq = seqs.gene_seq

    rna_emb = embedding_tools.get_embeddings_new('original_5_7.pkl', rna_seq, k=5, overlap=False)
    gene_emb = embedding_tools.get_embeddings_new('original_5_7.pkl', gene_seq, k=5, overlap=False)

    print(rna_emb)
    print(gene_emb)

