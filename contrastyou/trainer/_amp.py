from torch.cuda.amp import GradScaler, autocast


class AMPScalerMixin:

    def __init__(self, *, enable_scale: bool = False, accumulate_iter: int = 1, **kwargs) -> None:
        super(AMPScalerMixin, self).__init__(**kwargs)
        self.scaler = GradScaler(enabled=enable_scale)
        self._enable_scale = enable_scale
        self._accumulate_iter = accumulate_iter

    def scale_loss(self, loss):
        return self.scaler.scale(loss)

    def optimizer_step(self, optimizer, *, cur_iter: int):
        """this step updates the optimizer and the scaler in the same time."""
        if cur_iter % self._accumulate_iter == 0:
            self.scaler.step(optimizer)
            self.scaler.update()

    def optimizer_zero_grad(self, optimizer, *, cur_iter: int):
        if cur_iter % self._accumulate_iter == 0:
            optimizer.zero_grad()

    @property
    def use_mixed_train(self) -> bool:
        return self.scaler._enabled  # noqa

    @property
    def autocast(self):
        return autocast(enabled=self.use_mixed_train)

    @property
    def enable_scale(self) -> bool:
        return self._enable_scale

    @enable_scale.setter
    def enable_scale(self, value: bool):
        self._enable_scale = value
        self.scaler = GradScaler(enabled=value)
