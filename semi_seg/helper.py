from typing import Protocol


class SizedIterable(Protocol):
    def __len__(self):
        pass

    def __next__(self):
        pass

    def __iter__(self):
        pass
