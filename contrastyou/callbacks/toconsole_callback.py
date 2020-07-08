import sys

from tqdm import tqdm

from contrastyou.callbacks._callback import _EpochCallack, _TrainerCallback
from contrastyou.helper import flatten_dict, nice_dict


class TQDMCallback(_EpochCallack):

    def __init__(self, indicator_length=0, frequency_print=10) -> None:
        self._indicator_length = indicator_length
        self._frequency_print = frequency_print

    def before_run(self):
        self._indicator = tqdm(ncols=10, leave=True, dynamic_ncols=True)
        if self._indicator_length > 0:
            self._indicator = tqdm(total=self._indicator_length, ncols=10, leave=True, dynamic_ncols=True)
        self._n = 0

    def after_step(self, report_dict=None, *args, **kwargs):
        self._indicator.update(1)
        self._n += 1
        if self._n % self._frequency_print == 0:
            class_name = self._epocher.__class__.__name__
            current_epoch = self._epocher._cur_epoch
            report_dict = flatten_dict(report_dict)
            self._indicator.set_description(f"{class_name} Epoch {current_epoch:03d}")
            self._indicator.set_postfix(report_dict)
            sys.stdout.flush()

    def after_run(self, *args, **kwargs):
        self._indicator.close()


class PrintResultCallback(_EpochCallack, _TrainerCallback):
    def after_run(self, report_dict=None, *args, **kwargs):
        if report_dict:
            class_name = self._epocher.__class__.__name__
            cur_epoch = self._epocher._cur_epoch
            sys.stdout.flush()
            print(f"{class_name} Epoch {cur_epoch}: {nice_dict(flatten_dict(report_dict))}")
            sys.stdout.flush()

    def after_train(self, *args, **kwargs):
        storage = self._trainer._storage
        sys.stdout.flush()
        print(storage.summary())
        sys.stdout.flush()
