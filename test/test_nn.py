from unittest import TestCase

import torch
from torch import nn

from contrastyou.nn import ModuleBase, NoTrackable


class SmallModule(ModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(3, 3, 1)

        self._optimizer = torch.optim.Adam(self._conv1.parameters())


class TestModule(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._module1 = SmallModule()
        self._module2 = nn.Conv1d(1, 1, 1)

    def test_failed_case(self):
        self._module2.module1 = self._module1
        self._module1.module2 = self._module2

        with self.assertRaises(RecursionError):
            print(self._module2)
        with self.assertRaises(RecursionError):
            print(self._module1)
        with self.assertRaises(RecursionError):
            print(self._module1.state_dict())
        with self.assertRaises(RecursionError):
            print(self._module2.state_dict())

        with self.assertRaises(RecursionError):
            print(self._module1.to("cpu"))

        with self.assertRaises(RecursionError):
            print(self._module2.to("cpu"))

        with self.assertRaises(RecursionError):
            print(self._module1.apply(lambda x: x.to(torch.float)))

        with self.assertRaises(RecursionError):
            print(self._module1.state_dict())

    def test_useful_case(self):
        self._module2.module1 = NoTrackable(self._module1)
        self._module1.module2 = NoTrackable(self._module2)

        self._module1.module2 = self._module2

        print(self._module2)
        print(self._module1)
        print(self._module1.state_dict())
        print(self._module2.state_dict())

        print(self._module1.to("cpu"))

        print(self._module2.to("cpu"))

        print(self._module1.apply(lambda x: x.type(torch.float)))
        print(self._module1.state_dict().keys())
