# ======== base pretrain epocher mixin ================
import contextlib
import random
import typing as t
from abc import ABC, ABCMeta
from functools import partial
from pathlib import Path

import torch
from loguru import logger
from rising.utils.transforms import iter_transform

from contrastyou.utils import get_lrs_from_optimizer
from semi_seg.epochers.epocher import SemiSupervisedEpocher, assert_transform_freedom
from semi_seg.epochers.helper import preprocess_input_with_twice_transformation

if t.TYPE_CHECKING:
    from contrastyou.meters import MeterInterface

    _Base = SemiSupervisedEpocher
else:
    _Base = object


class _PretrainEpocherMixin(_Base, metaclass=ABCMeta):

    def __init__(self, *, chain_dataloader, inference_until: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._chain_dataloader = chain_dataloader
        self._inference_until = inference_until

    def configure_meters(self, meters: 'MeterInterface') -> 'MeterInterface':
        meter = super().configure_meters(meters)  # noqa
        meter.delete_meters(["sup_loss", "sup_dice"])
        return meter

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_implement(**kwargs)

    def _run_implement(self, **kwargs):
        for self.cur_batch_num, data in zip(self.indicator, self._chain_dataloader):
            seed = random.randint(0, int(1e7))
            (unlabeled_image, unlabeled_image_tf), _, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(data, self._device)

            unlabeled_file_path: t.List[str]
            file_path_root = Path(
                self._chain_dataloader._dataset.root_dir,  # noqa
                # self._chain_dataloader._dataset.folder_name,  # noqa
                self._chain_dataloader._dataset.mode,  # noqa
            )
            unlabeled_file_path = [(file_path_root / f).as_posix() for f in unlabeled_filename]

            unlabeled_image_tf = self.transform_with_seed(unlabeled_image_tf, mode="image", seed=seed)

            self.batch_update(
                cur_batch_num=self.cur_batch_num,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                seed=seed, unl_group=unl_group, unl_partition=unl_partition,
                unlabeled_filename=unlabeled_filename,
                unlabeled_file_path=unlabeled_file_path,
            )

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics2(report_dict, force_update=self.cur_batch_num == self.num_batches - 1)

    def _batch_update(self, *, cur_batch_num: int, unlabeled_image, unlabeled_image_tf, seed,  # type: ignore
                      unl_group, unl_partition, unlabeled_filename, **kwargs):
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)

        with self.autocast:
            unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            unlabeled_logits_tf = self.transform_with_seed(unlabeled_logits, seed=seed, mode="feature")

            reg_loss = self.regularization(
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                unlabeled_tf_logits=unlabeled_tf_logits,
                unlabeled_logits_tf=unlabeled_logits_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                unlabeled_filename=unlabeled_filename,
                affine_transformer=partial(self.transform_with_seed, seed=seed, mode="feature"),
                **kwargs
            )

        self.scale_loss(reg_loss).backward()
        self.optimizer_step(self._optimizer, cur_iter=cur_batch_num)

        with torch.no_grad():
            if self.meters:
                self.meters["reg_loss"].add(reg_loss.item())

    def _forward_pass(self, unlabeled_image, unlabeled_image_tf):  # noqa
        n_l, n_unl = 0, len(unlabeled_image)
        predict_logits = self._model(
            torch.cat([unlabeled_image, unlabeled_image_tf], dim=0), until=self._inference_until)
        unlabeled_logits, unlabeled_tf_logits = torch.split(predict_logits, (n_unl, n_unl), dim=0)
        return unlabeled_logits, unlabeled_tf_logits

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), None, filename, partition, group


_BaseInference = _PretrainEpocherMixin if t.TYPE_CHECKING else object


class _PretrainInferenceEpocherMixin(_BaseInference, metaclass=ABCMeta):
    def _batch_update(self, *, cur_batch_num: int, unlabeled_image, unlabeled_image_tf, seed,  # type: ignore # noqa
                      unl_group, unl_partition, unlabeled_filename):  # noqa

        with self.autocast:
            unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            unlabeled_logits_tf = self.transform_with_seed(unlabeled_logits, seed=seed, mode="feature")

            reg_loss = self.regularization(
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                unlabeled_tf_logits=unlabeled_tf_logits,
                unlabeled_logits_tf=unlabeled_logits_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                unlabeled_filename=unlabeled_filename,
                affine_transformer=partial(self.transform_with_seed, seed=seed, mode="feature")
            )
        # remove update
        with torch.no_grad():
            self.meters["reg_loss"].add(reg_loss.item())  # type: ignore

    @contextlib.contextmanager
    def disable_rising_augmentation(self):
        logger.trace("disable rising augmentation")
        rising_wrapper = self._affine_transformer
        geometric = iter_transform(rising_wrapper.geometry_transform)
        geometric_p = {id(x): x.p for x in geometric if hasattr(x, "p")}

        intensity = iter_transform(rising_wrapper.intensity_transform)
        intensity_p = {id(x): x.p for x in intensity if hasattr(x, "p")}

        for t in iter_transform(rising_wrapper.geometry_transform):
            if hasattr(t, "p"):
                setattr(t, "p", 0)
        for t in iter_transform(rising_wrapper.intensity_transform):
            if hasattr(t, "p"):
                setattr(t, "p", 0)
        yield
        logger.trace("resume rising augmentation")

        for t in iter_transform(rising_wrapper.geometry_transform):
            if hasattr(t, "p"):
                setattr(t, "p", geometric_p[id(t)])
        for t in iter_transform(rising_wrapper.intensity_transform):
            if hasattr(t, "p"):
                setattr(t, "p", intensity_p[id(t)])

    def _run_implement(self, **kwargs):
        with self.disable_rising_augmentation(), torch.no_grad():
            return super(_PretrainInferenceEpocherMixin, self)._run_implement(**kwargs)


class PretrainEncoderEpocher(_PretrainEpocherMixin, SemiSupervisedEpocher, ABC):
    def _assertion(self):
        assert_transform_freedom(self._labeled_loader, True)
        if self._unlabeled_loader is not None:
            assert_transform_freedom(self._unlabeled_loader, True)


class PretrainDecoderEpocher(_PretrainEpocherMixin, SemiSupervisedEpocher, ABC):
    def _assertion(self):
        assert_transform_freedom(self._labeled_loader, False)
        if self._unlabeled_loader is not None:
            assert_transform_freedom(self._unlabeled_loader, False)


class PretrainDecoderEpocherInference(_PretrainInferenceEpocherMixin, _PretrainEpocherMixin,
                                      SemiSupervisedEpocher, ABC):
    def _assertion(self):
        assert_transform_freedom(self._labeled_loader, False)
        if self._unlabeled_loader is not None:
            assert_transform_freedom(self._unlabeled_loader, False)
