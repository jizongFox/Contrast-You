from unittest import TestCase

import torch
from torch import nn

from contrastyou.nn import ModuleBase, NoTrackable, Buffer


class SmallModule(ModuleBase):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(3, 3, 1)
        self.name = Buffer(name)

        self._optimizer = torch.optim.Adam(self._conv1.parameters())


class TestModule(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._module1 = SmallModule("model1")
        self._module2 = SmallModule("model2")

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

    def test_assign_value(self):
        self._module1.value1 = Buffer(1)
        assert "value1" in self._module1._persist_buffer
        self._module1.value1 = 23
        assert self._module1._persist_buffer["value1"] == 23
        assert "value1" not in self._module1.__dict__
