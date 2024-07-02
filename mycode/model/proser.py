import torch
from torch import nn


class DummyCls(nn.Module):
    def __init__(self, head, C):
        super().__init__()
        self.head = head

        in_dim = head.in_features
        self.dummy_cls = nn.Linear(in_dim, C)

    def forward(self, x):
        # Nx2048
        logits1 = self.head(x)  # Nx200
        logits2, _ = self.dummy_cls(x).max(dim=-1, keepdim=True)  # NxC -> Nx1
        if len(logits1.shape) == 3:
            logits = torch.concatenate([logits1, logits2], dim=2)
        else:
            logits = torch.concatenate([logits1, logits2], dim=1)
        return logits


def get_openset_head(head, C, gpu):
    new_head = DummyCls(head, C)
    return new_head.cuda(gpu)


def get_original_head(head: DummyCls):
    return head.head


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa = nn.Linear(10, 10)
        self.head = nn.Linear(10, 3)

    def forward(self, x):
        return self.head(self.aa(x))


if __name__ == '__main__':
    from torch.optim import SGD

    torch.manual_seed(100)

    net = Net()
    original_head = net.head
    open_head = get_openset_head(original_head, 2)

    optimizer = SGD(open_head.parameters(), lr=1e-3)

    net.head = original_head

    print(original_head.weight)

    data = torch.rand((3, 10))
    out = net(data)
    out = out.sum()
    out.backward()
    optimizer.step()

    print(original_head.weight)

    net.head = open_head

    print(open_head.head.weight)

    data = torch.rand((3, 10))
    out = net(data)
    out = out.sum()
    out.backward()
    optimizer.step()

    print(open_head.head.weight)
    print(original_head.weight)