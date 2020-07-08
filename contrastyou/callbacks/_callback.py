import weakref


class EpochCallBacks:

    def __init__(self, train_callbacks=None, val_callbacks=None, test_callbacks=None) -> None:
        self._train_callbacks = train_callbacks
        self._val_callbacks = val_callbacks
        self._test_callbacks = test_callbacks
        if train_callbacks:
            for c in self._train_callbacks:
                assert isinstance(c, _EpochCallack), c
        if val_callbacks:
            for c in self._val_callbacks:
                assert isinstance(c, _EpochCallack), c
        if test_callbacks:
            for c in self._test_callbacks:
                assert isinstance(c, _EpochCallack), c


class _EpochCallack:
    """
    callback for epocher
    """

    def set_epocher(self, epocher):
        self._epocher = weakref.proxy(epocher)

    def before_run(self):
        pass

    def after_run(self, *args, **kwargs):
        pass

    def before_step(self):
        pass

    def after_step(self, *args, **kwargs):
        pass


class _TrainerCallback:
    """
    callbacks for trainer
    """

    def set_trainer(self, trainer):
        self._trainer = weakref.proxy(trainer)

    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass
