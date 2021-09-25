import time
from pprint import pprint

import torch

from contrastyou.meters import UniversalDice
from contrastyou.meters import UniversalDice as UniversalDice2


class timer:

    def __init__(self) -> None:
        super().__init__()

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"elapsed time: {self.end - self.start: .3f}")


N = 10000
device = "cuda"
dtype = torch.half

prediction = torch.randn(10, 4, 256, 256, device=device, dtype=dtype).max(1)[1]
target = torch.randn(10, 4, 256, 256, device=device, dtype=dtype).max(1)[1]

meter1 = UniversalDice(C=4)
meter2 = UniversalDice2(C=4)
with timer():
    for i in range(N):
        pred = torch.randn(10, 4, 256, 256, device=device, dtype=dtype)
        group_name = str(i)
        prediction = pred.max(1)[1]
        target = (pred + torch.randn_like(pred) * 0.9).max(1)[1]
        meter1.add(prediction, target, group_name=group_name)
        # meter2.add(prediction, target, group_name=group_name)

    result1 = meter1.summary()
    # result2 = meter2.summary()
pprint(result1)
# pprint(result2)
#
with timer():
    for i in range(N):
        pred = torch.randn(10, 4, 256, 256, device=device, dtype=dtype)
        group_name = str(i)

        prediction = pred.max(1)[1]
        target = (pred + torch.randn_like(pred) * 0.1).max(1)[1]
        # meter1.add(prediction, target)
        meter2.add(prediction, target, group_name=group_name)

    # result1 = meter1.summary()
    result2 = meter2.summary()
# pprint(result1)
pprint(result2)
