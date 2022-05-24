import torch
from torchmetrics.functional import auc
x = torch.tensor([1.0, .0, .0])
y = torch.tensor([0, 0, 1])
auc(x, y)
auc(x, y, reorder=True)


from torchmetrics.functional import auroc
preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
target = torch.tensor([0, 0, 1, 1, 1])
auroc(x, y, pos_label=1)
tensor(0.5000)