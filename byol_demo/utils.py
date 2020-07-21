from torch.nn import functional as F


class TransTwice:

    def __init__(self, transform) -> None:
        super().__init__()
        self._transform = transform

    def __call__(self, img):
        return [self._transform(img), self._transform(img)]


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
