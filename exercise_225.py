import pandas as pd
import torch

data = pd.read_csv("abalone.data", header=None, skiprows=1)


# print(type(data[0]))


df = pd.DataFrame(
    data,
    columns=[
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ],
)

print(df)
