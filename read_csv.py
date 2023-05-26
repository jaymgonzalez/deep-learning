import pandas as pd
import torch

data = pd.read_csv("house_tiny.csv")
print(data)


inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.fillna(inputs.mean())
print(inputs)

X, y = torch.tensor(inputs.values), torch.tensor(targets.values)

print(X, y)
