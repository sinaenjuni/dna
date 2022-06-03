import torch
import torch.nn as nn
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





input1 = torch.tensor([[1.,1.]])
input2 = torch.tensor([[1.,0.]])

inputs = torch.cat((input1, input2), dim=0)
print(inputs.size())
print(inputs.unsqueeze(0).shape, inputs.unsqueeze(1).shape)
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
output = cos(inputs.unsqueeze(0), inputs.unsqueeze(1))
print(output.shape)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output.shape)



input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output.shape)

inputs = torch.cat((input1, input2), dim=0)
print(inputs.unsqueeze(0).shape, inputs.unsqueeze(1).shape)
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
output = cos(inputs.unsqueeze(0), inputs.unsqueeze(1))
print(output.shape)

