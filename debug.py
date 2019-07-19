import torch.nn as nn
import torch.tensor as tt

ct = nn.CrossEntropyLoss()

label = tt([[1, 1], [2, 0]])

pred = tt([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])


print(label.view(-1))

loss = ct(pred, label.view(-1))

print(loss)