from torch.cuda.amp import GradScaler, autocast


class AMPScaler:

    def __init__(self, *, scaler: GradScaler, accumulate_iter: int = 1) -> None:
        self.scaler = scaler
        assert accumulate_iter >= 1
        self._accumulate_iter = accumulate_iter

    def scale_loss(self, loss):
        return self.scaler.scale(loss / self._accumulate_iter)

    def optimizer_step(self, optimizer, *, cur_iter: int):
        """this step updates the optimizer and the scaler in the same time."""
        if cur_iter % self._accumulate_iter == 0:
            self.scaler.step(optimizer)
            self.scaler.update()

    def optimizer_zero(self, optimizer, *, cur_iter: int):
        if cur_iter % self._accumulate_iter == 0:
            optimizer.zero_grad()

    @property
    def use_mixed_train(self) -> bool:
        return self.scaler._enabled  # noqa

    @property
    def autocast(self):
        return autocast(enabled=self.use_mixed_train)
