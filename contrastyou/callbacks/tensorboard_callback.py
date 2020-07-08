from contrastyou.callbacks._callback import _TrainerCallback
from contrastyou.helper import flatten_dict
from contrastyou.trainer._trainer import EpochResult
from contrastyou.writer import SummaryWriter


class SummaryCallback(_TrainerCallback):
    def __init__(self, log_dir=None) -> None:
        self._writer = SummaryWriter(log_dir)

    def after_epoch(self, epoch_result: EpochResult = None, *args, **kwargs):

        current_epoch = self._trainer._cur_epoch
        if epoch_result.train_result:
            self._writer.add_scalar_with_tag(tag="tra", tag_scalar_dict=flatten_dict(epoch_result.train_result),
                                             global_step=current_epoch)

        if epoch_result.val_result:
            self._writer.add_scalar_with_tag(tag="val", tag_scalar_dict=flatten_dict(epoch_result.val_result),
                                             global_step=current_epoch)

        if epoch_result.test_result:
            self._writer.add_scalar_with_tag(tag="test", tag_scalar_dict=flatten_dict(epoch_result.test_result),
                                             global_step=current_epoch)
