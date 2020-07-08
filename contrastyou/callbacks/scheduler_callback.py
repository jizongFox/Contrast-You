from ._callback import _TrainerCallback


class SchedulerCallback(_TrainerCallback):

    def after_epoch(self, *args, **kwargs):
        scheduler = self._trainer._model.scheduler
        if scheduler:
            scheduler.step()
