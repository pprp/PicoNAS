import torch
import torch.nn as nn


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 4)
        self.l2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.l2(self.l1(x))


class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, y, params):
        return params * self.loss_fn(x, y)


if __name__ == '__main__':
    x = torch.randn(4, 3)
    y = torch.randn(5)

    m = TestModel()
    param = nn.Parameter(torch.randn(1))
    o = m(x)

    loss_fn = CustomLoss()
    loss = loss_fn(o, y, param)

    model_params = tuple(m.parameters())

    d_model = torch.autograd.grad(loss, model_params, retain_graph=True)
    d_param = torch.autograd.grad(loss, param)
    print(d_param)
