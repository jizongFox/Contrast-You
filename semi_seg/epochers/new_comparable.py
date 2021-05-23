import random

import torch
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.utils import class2one_hot
from loguru import logger

from contrastyou.utils import get_dataset
from . import preprocess_input_with_twice_transformation
from .new_epocher import SemiSupervisedEpocher


class MixupEpocher(SemiSupervisedEpocher):
    meter_focus = "mixup"

    def _assertion(self):
        labeled_set = get_dataset(self._labeled_loader)
        labeled_transform = labeled_set.transforms
        assert labeled_transform._total_freedom is True  # noqa

        if self._unlabeled_loader is not None:
            unlabeled_set = get_dataset(self._unlabeled_loader)
            unlabeled_transform = unlabeled_set.transforms
            assert unlabeled_transform._total_freedom is True  # noqa

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_mix_up(**kwargs)

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), (target, target_ct), filename, partition, group

    def _run_mix_up(self, **kwargs):
        for self.cur_batch_num, labeled_data, in zip(self.indicator, self._labeled_loader):
            seed = random.randint(0, int(1e7))
            (labeled_image, labeled_image_tf), (labeled_target, labeled_target_tf), labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)

            if self.cur_batch_num < 5:
                logger.trace(f"{self.__class__.__name__}--"
                             f"cur_batch:{self.cur_batch_num}, labeled_filenames: {','.join(labeled_filename)}")

            label_logits = self.forward_pass(
                labeled_image=labeled_image,
                labeled_image_tf=labeled_image_tf
            )

            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
            # regularized part
            reg_loss = self.regularization(
                labeled_image=labeled_image,
                labeled_image_tf=labeled_image_tf,
                labeled_target=labeled_target,
                labeled_target_tf=labeled_target_tf,
            )

            total_loss = sup_loss + reg_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                                group_name=label_group)
                    self.meters["reg_loss"].add(reg_loss.item())

                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict, cache_time=10)

    def _forward_pass(self, labeled_image, **kwargs):
        label_logits = self._model(labeled_image)
        return label_logits
