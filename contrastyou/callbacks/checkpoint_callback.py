from ._callback import _TrainerCallback
from ..trainer._trainer import EpochResult
class SaveBestCheckpoint(_TrainerCallback):



    def after_epoch(self, epoch_result:EpochResult, *args, **kwargs):
        self._save_dir = self._trainer._save_dir
        assert self._save_dir.exists()
        cur_score = epoch_result.val_result["confusion_mx"]["acc"]
        self._trainer._save_checkpoint(cur_score)



