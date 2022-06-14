from functools import lru_cache

from loguru import logger
from torch.cuda.amp import GradScaler, autocast


@lru_cache(maxsize=1)
def _warning(accumulate_iter):
    logger.warning(
        f"._accumulate_iter={accumulate_iter} > 1, may reduce performance.")


class AMPScaler:

    def __init__(self, *, scaler: GradScaler, accumulate_iter: int = 1) -> None:
        self.scaler = scaler
        assert accumulate_iter >= 1
        self._accumulate_iter = accumulate_iter
        if self._accumulate_iter > 1:
            _warning(self._accumulate_iter)

    def scale_loss(self, loss):
        return self.scaler.scale(loss / self._accumulate_iter)

    def optimizer_step(self, optimizer, *, cur_iter: int):
        """this step updates the optimizer and the scaler in the same time."""
        if cur_iter % self._accumulate_iter == (self._accumulate_iter - 1):
            self.scaler.step(optimizer)
            self.scaler.update()
            if self._accumulate_iter > 1 and cur_iter <= 10:
                logger.trace(f"iter: {cur_iter}, step optimizer")

    def optimizer_zero(self, optimizer, *, cur_iter: int):
        if cur_iter % self._accumulate_iter == 0:
            optimizer.zero_grad()
            if self._accumulate_iter > 1 and cur_iter <= 10:
                logger.trace(f"iter: {cur_iter}, zero optimizer")

    @property
    def use_mixed_train(self) -> bool:
        return self.scaler._enabled  # noqa

    @property
    def autocast(self):
        return autocast(enabled=self.use_mixed_train)
