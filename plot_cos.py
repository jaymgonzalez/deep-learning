import numpy as np
from d2l import torch as d2l
import torch
import matplotlib as mpl

x = np.linspace(-np.pi, np.pi, 100)
x = torch.tensor(x, requires_grad=True)
y = torch.sin(x)
for i in range(100):
    y[i].backward(retain_graph=True)

d2l.plot(x.detach(), (y.detach(), x.grad), legend=(("sin(x)", "grad w.s.t x")))
mpl.pyplot.show()
