from ._callback import _TrainerCallback
from ..trainer._trainer import EpochResult


class StorageCallback(_TrainerCallback):
    def after_epoch(self, epoch_result: EpochResult = None, *args, **kwargs):
        if epoch_result:
            storage = self._trainer._storage
            if epoch_result.train_result:
                storage.put_all({"tra": epoch_result.train_result})
            if epoch_result.val_result:
                storage.put_all({"val": epoch_result.val_result})
            if epoch_result.test_result:
                storage.put_all({"test": epoch_result.test_result})
            storage = None
