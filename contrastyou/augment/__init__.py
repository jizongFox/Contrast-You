class SequentialWrapperTwice:
    def __init__(self, transform=None) -> None:
        self._transform = transform

    def __call__(
        self, *imgs, random_seed=None
    ):
        return [
            self._transform.__call__(*imgs, random_seed=random_seed),
            self._transform.__call__(*imgs, random_seed=random_seed),
        ]
