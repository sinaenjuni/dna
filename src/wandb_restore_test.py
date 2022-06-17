
import wandb
api = wandb.Api()

sweep = api.sweep("sinaenjuni/dna-protein2vec/0i51p4g9")
sweep = api.sweep("sinaenjuni/dna-protein2vec/xtzabb5e")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_auc", 0), reverse=True)
val_auc = runs[0].summary.get("val_auc", 0)


print(f"Best run {runs[0].name} with {val_auc}% validation accuracy")

runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")


