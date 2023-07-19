import os
import random
import typing as t
import warnings
import weakref
from abc import ABCMeta
from itertools import chain

import torch
from loguru import logger
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from contrastyou.arch._base import _Network  # noqa
from contrastyou.arch.utils import SingleFeatureExtractor
from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.losses.cross_correlation import CCLoss
from contrastyou.losses.discreteMI import IIDSegmentationLoss
from contrastyou.losses.kl import Entropy, KL_div
from contrastyou.losses.redundancy_reduction import RedundancyCriterion
from contrastyou.meters import AverageValueMeter
from contrastyou.projectors import CrossCorrelationProjector
from contrastyou.utils import class_name, item2str, probs2one_hot, fix_all_seed_within_context, simplex, deprecated
from contrastyou.writer import get_tb_writer
from semi_seg.hooks.utils import FeatureMapSaver, DistributionTracker, joint_2D_figure, MatrixSaver
from skimage import io as sio
from pathlib import Path

if t.TYPE_CHECKING:
    from contrastyou.projectors.nn import _ProjectorHeadBase  # noqa
    from contrastyou.meters import MeterInterface
    from contrastyou.losses.discreteMI import IMSATLoss, IMSATDynamicWeight

__all__ = ["ProjectorGeneralHook", "_CrossCorrelationHook",
           "_MIHook"]


# new interface
class _TinyHook(metaclass=ABCMeta):

    def __init__(self, *, name: str, criterion: nn.Module, weight: float) -> None:
        self.name = name
        self.criterion = criterion
        self.weight = weight
        self.meters = None
        self.hook = None
        logger.trace(f"Created {self.__class__.__name__}:{self.name} with weight={self.weight}.")

    def configure_meters(self, meters: 'MeterInterface'):
        meters.register_meter(self.name, AverageValueMeter())
        return meters

    def __call__(self, **kwargs) -> Tensor:
        loss = self.criterion(**kwargs)
        if self.meters:
            self.meters[self.name].add(loss.item())
        return loss * self.weight

    def close(self):
        pass

    @staticmethod
    def get_tb_writer():
        return get_tb_writer()

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.name}{self.__repr_extra__()}"

    def __repr_extra__(self):
        return f"weight={self.weight}"


class ProjectorGeneralHook(TrainerHook):

    def __init__(self, *, name: str, model: _Network, feature_name: str,
                 projector_params: t.Dict[str, t.Any], save: bool = False):
        super().__init__(hook_name=name)
        self._feature_name = feature_name
        logger.info(
            f"Creating {class_name(self)} @{feature_name}.")
        self._extractor = SingleFeatureExtractor(
            model=model, feature_name=feature_name  # noqa
        )
        input_dim = model.get_channel_dim(feature_name)
        logger.trace(f"Creating projector with {item2str(projector_params)}")
        with logger.contextualize(enabled=False):
            self._projector = CrossCorrelationProjector(input_dim=input_dim, **projector_params)

        self._feature_hooks = []
        self._dist_hooks = []
        self.save = save
        self.saver = None
        self.dist_saver = None

    def after_initialize(self):
        if self.save:
            self.saver = FeatureMapSaver(save_dir=self.trainer.absolute_save_dir,
                                         folder_name=f"vis/{self._hook_name}")
            self.dist_saver = DistributionTracker(save_dir=self.trainer.absolute_save_dir,
                                                  folder_name=f"dist/{self._hook_name}")
            self.matrix_saver = MatrixSaver(self.trainer.absolute_save_dir, f"matrix/{self._hook_name}")

    def register_feat_hook(self, *hook: '_TinyHook'):
        logger.debug(f"register {','.join([str(x) for x in hook])}")
        self._feature_hooks.extend(hook)

    def register_dist_hook(self, *hook: '_TinyHook'):
        logger.debug(f"register {','.join([str(x) for x in hook])}")
        self._dist_hooks.extend(hook)

    def __call__(self, **kwargs):
        if (len(self._feature_hooks) + len(self._dist_hooks)) == 0:
            raise RuntimeError(f"hooks not registered for {class_name(self)}.")

        return _ProjectorEpocherGeneralHook(
            name=self._hook_name, extractor=self._extractor, projector=self._projector, dist_hooks=self._dist_hooks,
            feat_hooks=self._feature_hooks, saver=self.saver, dist_saver=self.dist_saver, matrix_saver=self.matrix_saver
        )

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return [self._projector, ]

    def close(self):
        if self.save:
            self.saver.zip()
            self.dist_saver.zip()


# try to convert this to hook
class _ProjectorEpocherGeneralHook(EpocherHook):

    def __init__(self, *, name: str, extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 dist_hooks: t.Sequence[_TinyHook] = (), feat_hooks: t.Sequence[_TinyHook] = (),
                 saver: FeatureMapSaver = None,
                 dist_saver: DistributionTracker = None,
                 matrix_saver: MatrixSaver = None) -> None:
        super().__init__(name=name)
        self.extractor = extractor
        self.extractor.bind()
        self.projector = projector
        self._feature_hooks = feat_hooks
        self._dist_hooks = dist_hooks
        self.saver = saver
        self.dist_saver = dist_saver
        self.mx_saver = matrix_saver

    def configure_meters_given_epocher(self, meters: 'MeterInterface'):
        meters = super(_ProjectorEpocherGeneralHook, self).configure_meters_given_epocher(meters)
        for h in chain(self._dist_hooks, self._feature_hooks):
            h.meters = meters
            h.hook = weakref.proxy(self)
            h.configure_meters(meters)
        return meters

    def before_forward_pass(self, **kwargs):
        self.extractor.clear()
        self.extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self.extractor.set_enable(False)

    def _call_implementation(self, unlabeled_image_tf: Tensor, unlabeled_logits_tf: Tensor,
                             affine_transformer: t.Callable[[Tensor], Tensor],
                             unlabeled_image: Tensor, seed: int, **kwargs):
        cur_epoch = self.epocher.cur_epoch
        cur_batch_num = self.epocher.cur_batch_num

        save_image_condition = cur_batch_num == 0 and cur_epoch % 3 == 0 and self.saver is not None
        save_image_condition = save_image_condition or (self.save_flag and self.saver is not None)
        save_matrix_condition = self.save_np_flag and (self.mx_saver is not None)

        n_unl = len(unlabeled_logits_tf)
        feature_ = self.extractor.feature()[-n_unl * 2:]
        _unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(_unlabeled_features)
        with fix_all_seed_within_context(seed):
            feature_loss = self._run_feature_hooks(
                input1=unlabeled_features_tf,
                input2=unlabeled_tf_features,
                image=unlabeled_image_tf
            )
        if save_image_condition and self.saver is not None:
            self.saver.save_map(
                image=unlabeled_image_tf, feature_map1=unlabeled_tf_features, feature_map2=unlabeled_features_tf,
                cur_epoch=cur_epoch, cur_batch_num=cur_batch_num, save_name="feature"
            )
        if save_matrix_condition:
            self.mx_saver.save_matrix(matrix=unlabeled_tf_features, cur_epoch=cur_epoch, cur_batch_num=cur_batch_num,
                                      save_name="feature")

        projected_dist_tf, projected_tf_dist = zip(*[torch.chunk(x, 2) for x in self.projector(
            torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0))])

        if save_image_condition:
            if self.saver is not None:
                self.saver.save_map(
                    image=unlabeled_image_tf, feature_map1=projected_dist_tf[0], feature_map2=projected_tf_dist[0],
                    cur_epoch=cur_epoch, cur_batch_num=cur_batch_num, save_name="probability"
                )
            if self.dist_saver is not None:
                self.dist_saver.save_map(dist1=projected_dist_tf[0], dist2=projected_tf_dist[0],
                                         cur_epoch=cur_epoch)

        if save_matrix_condition:
            self.mx_saver.save_matrix(matrix=projected_dist_tf[0], cur_epoch=cur_epoch, cur_batch_num=cur_batch_num,
                                      save_name="probability")

        with fix_all_seed_within_context(seed):
            dist_losses = tuple(
                self._run_dist_hooks(
                    input1=prob1, input2=prob2, image=unlabeled_image_tf,
                    feature_map1=unlabeled_tf_features, feature_map2=unlabeled_features_tf, saver=self.saver,
                    save_image_condition=save_image_condition, cur_epoch=cur_epoch,
                    cur_batch_num=cur_batch_num, batch_data=kwargs["batch_data"], target_tf = affine_transformer(kwargs["batch_data"]["gt"][0].float()), unlabeled_filename=kwargs["unlabeled_filename"]
                )
                for prob1, prob2 in zip(projected_tf_dist, projected_dist_tf)
            )
        dist_loss = sum(dist_losses) / len(dist_losses)

        return feature_loss + dist_loss

    def _run_feature_hooks(self, **kwargs):
        return sum(h(**kwargs) for h in self._feature_hooks)

    def _run_dist_hooks(self, **kwargs):
        return sum(h(**kwargs) for h in self._dist_hooks)

    def close(self):
        self.extractor.remove()
        for h in chain(self._feature_hooks, self._dist_hooks):
            h.close()

    @property
    def save_flag(self) -> bool:
        return os.environ.get("contrast_save_flag", "false") == "true"

    @property
    def save_np_flag(self) -> bool:
        return os.environ.get("contrast_save_np_flag", "false") == "true"


class _CrossCorrelationHook(_TinyHook):

    def __init__(self, *, name: str = "cc", weight: float, kernel_size: int, diff_power: float = 0.75) -> None:
        criterion = CCLoss(win=(kernel_size, kernel_size))
        super().__init__(name=name, criterion=criterion, weight=weight)
        self._ent_func = Entropy(reduction="none")
        self._diff_power = diff_power

    def __repr_extra__(self):
        return f"{super(_CrossCorrelationHook, self).__repr_extra__()} diff_power={self._diff_power}"

    # force not using amp mixed precision training.
    @autocast(enabled=False)
    def __call__(self, *, image: Tensor, input1: Tensor, input2: Tensor, saver: "FeatureMapSaver",
                 save_image_condition: bool, cur_epoch: int, cur_batch_num: int, **kwargs):
        device = image.device
        self.criterion.to(device)  # noqa

        losses, self.diff_image, self.diff_prediction = zip(*[
            self.cc_loss_per_head(image=image, predict_simplex=x) for x in
            chain([input1, input2])
        ])
        loss = sum(losses) / len(losses)
        if self.meters:
            self.meters[self.name].add(loss.item())

        if save_image_condition:
            saver.save_map(
                image=self.diff_image[0], feature_map1=self.diff_prediction[0],
                feature_map2=self.diff_prediction[1],
                cur_epoch=cur_epoch, cur_batch_num=cur_batch_num,
                save_name="cross_correlation", feature_type="image"
            )

        return loss * self.weight

    def norm(self, image: Tensor, min=0.0, max=1.0, slicewise=True):
        return torch.stack([self._norm(x) for x in image], dim=0) if slicewise else self._norm(image, min, max)

    def _norm(self, image: Tensor, min=0.0, max=1.0):
        min_, max_ = image.min().detach(), image.max().detach()
        image = image - min_
        image = image / (max_ - min_ + 1e-6)
        return image * (max - min) + min

    @staticmethod
    def diff(image: Tensor):
        assert image.dim() == 4
        dx = image - torch.roll(image, shifts=1, dims=2)
        dy = image - torch.roll(image, shifts=1, dims=3)
        d = torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.mean(d, dim=1, keepdims=True)  # noqa

    def cc_loss_per_head(self, image: Tensor, predict_simplex: Tensor):
        if tuple(image.shape[-2:]) != tuple(predict_simplex.shape[-2:]):
            h, w = predict_simplex.shape[-2:]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                image = F.interpolate(image, size=(h, w), mode="bilinear")
        # the diff power applies only on edges.
        diff_image = self.norm(self.diff(image), min=0, max=1).pow(self._diff_power)
        diff_tf_softmax = self.norm(self._ent_func(predict_simplex), min=0, max=1).unsqueeze(1)

        loss = self.criterion(
            diff_tf_softmax,
            diff_image
        )
        return loss, diff_image, diff_tf_softmax


class _MIHook(_TinyHook):

    def __init__(self, *, name: str = "mi", weight: float, lamda: float, padding: int = 0, symmetric=True) -> None:
        criterion: IIDSegmentationLoss = IIDSegmentationLoss(lamda=lamda, padding=padding, symmetric=symmetric)
        self.lamda = lamda
        self.padding = padding
        self.symmetric = symmetric
        super().__init__(name=name, criterion=criterion, weight=weight)

    def __repr_extra__(self):
        return super(_MIHook, self).__repr_extra__() + \
               f" lamda={self.lamda} padding={self.padding} symmetric={self.symmetric}"

    def __call__(self, input1: Tensor, input2: Tensor, cur_epoch: int, **kwargs):
        if self.weight == 0:
            if self.meters:
                self.meters[self.name].add(0)
            return torch.tensor(0, device=input1.device, dtype=input1.dtype)
        loss = self.criterion(input1, input2)
        if self.meters:
            self.meters[self.name].add(loss.item())

        if kwargs.get("save_image_condition", False):
            self.criterion: IIDSegmentationLoss
            joint_2D_figure(self.criterion.get_joint_matrix(), tb_writer=self.get_tb_writer(), cur_epoch=cur_epoch,
                            tag=f"{class_name(self)}_{self.name}")

        return loss * self.weight


class _RedundancyReduction(_TinyHook):

    def __init__(self, *, name: str = "rr", weight: float, symmetric: bool = True, lamda: float = 1, alpha: float,
                 ) -> None:
        self.lamda = lamda
        self.symmetric = symmetric
        self.alpha = alpha
        criterion = RedundancyCriterion(symmetric=symmetric, lamda=lamda, alpha=alpha)
        super().__init__(name=name, criterion=criterion, weight=weight)

    def __repr_extra__(self):
        return super(_RedundancyReduction, self).__repr_extra__() + \
               f" lamda={self.lamda} alpha={self.alpha} symmetric={self.symmetric}"

    def __call__(self, input1: Tensor, input2: Tensor, cur_epoch: int, batch_data=None, **kwargs):
        if self.weight == 0:
            if self.meters:
                self.meters[self.name].add(0)
            return torch.tensor(0, device=input1.device, dtype=input1.dtype)

        self.criterion: RedundancyCriterion
        # cur_mixed_ratio: 0: IIC
        # 1: Barlow-twin.
        # cur_mixed_ratio = min(float(cur_epoch / self.max_epoch), 0.2)
        # self.criterion.set_ratio(cur_mixed_ratio)
        loss = self.criterion(input1, input2)
        if self.meters:
            self.meters[self.name].add(loss.item())

        if kwargs.get("save_image_condition", False):
            self.criterion: RedundancyCriterion
            joint_2D_figure(self.criterion.get_joint_matrix(), tb_writer=self.get_tb_writer(), cur_epoch=cur_epoch,
                            tag=f"{class_name(self)}_{self.name}")
            
        if batch_data is not None and "target_tf" in kwargs and os.environ.get("SAVE_MAPPING", "0")=="1":
            target_tf = kwargs["target_tf"]
            prob = input1
            unlabeled_filename = kwargs["unlabeled_filename"]
            
            save_dir = os.environ.get("SAVE_DIR", None)
            assert save_dir is not None   
            Path(save_dir, "predict").mkdir(exist_ok=True, parents=True)
            Path(save_dir, "gt").mkdir(exist_ok=True, parents=True)
            
            def save_image(predict:Tensor, target:Tensor, filename:str):
                import numpy as np
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sio.imsave(Path(save_dir, "predict", filename+".png"), predict.argmax(0).cpu().numpy().astype(np.uint8))
                    sio.imsave(Path(save_dir, "gt", filename+".png"), target.squeeze().cpu().numpy().astype(np.uint8))
                
                
            for cur_pred, cur_target, cur_filename in zip(prob, target_tf, unlabeled_filename):
                save_image(cur_pred, cur_target, cur_filename)
                     

        return loss * self.weight


class _CenterCompactnessHook(_TinyHook):
    @deprecated
    def __init__(self, *, name: str = "center", weight: float) -> None:
        super().__init__(name=name, criterion=None, weight=weight)  # noqa
        del self.criterion

    def __call__(self, input1, input2, feature_map1, feature_map2, image, **kwargs):
        if self.weight == 0:
            if self.meters:
                self.meters[self.name].add(0)
            return torch.tensor(0, dtype=input1.dtype, device=input1.device)
        if torch.rand((1,)).item() > 0.5:
            return torch.tensor(0, dtype=input1.dtype, device=input1.device)

        if torch.rand((1,)).item() > 0.5:
            loss = self.forward(probability_simplex=input1, feature_map=feature_map1)
        else:
            loss = self.forward(probability_simplex=input2, feature_map=feature_map2)
        if self.meters:
            self.meters[self.name].add(loss.item())
        return loss * self.weight

    def forward(self, probability_simplex: Tensor, feature_map: Tensor):
        dims = probability_simplex.shape[1]
        one_hot_mask = probs2one_hot(probability_simplex, class_dim=1)
        losses = []
        # randomly choose some dimension
        for dim in random.sample(range(dims), max(5, dims // 5)):
            cur_mask = one_hot_mask[:, dim].bool().unsqueeze(1)
            if cur_mask.sum() == 0:
                continue
            prototype = self.masked_average_pooling(feature_map, cur_mask)
            loss = self.center_loss(feature=feature_map, mask=cur_mask, prototype=prototype)
            losses.append(loss)
        if len(losses) == 0:
            return torch.tensor(0, dtype=probability_simplex.dtype, device=probability_simplex.device)

        return sum(losses) / len(losses)

    @staticmethod
    def masked_average_pooling(feature: Tensor, mask: Tensor):
        b, c, *_ = feature.shape
        used_feature = feature.swapaxes(1, 0).masked_select(mask.swapaxes(1, 0)).reshape(c, -1)
        return used_feature.mean(dim=1).reshape(1, c, 1, 1)

    def center_loss(self, feature: Tensor, mask: Tensor, prototype: Tensor):
        return torch.mean((feature - prototype).pow(2), dim=1, keepdim=True).masked_select(mask).mean()


class _IMSATHook(_TinyHook):

    def __init__(self, *, name: str = "imsat", weight: float, use_dynamic=True, lamda: float = 1.0) -> None:
        criterion = IMSATDynamicWeight(use_dynamic=use_dynamic, lamda=lamda)
        super().__init__(name=name, criterion=criterion, weight=weight)
        self.lamda = lamda
        self.use_dynamic = use_dynamic

    def __repr_extra__(self):
        return super(_IMSATHook, self).__repr_extra__() + \
               f" lamda={self.lamda} dynamic={self.use_dynamic}"

    def configure_meters(self, meters: 'MeterInterface'):
        meters = super().configure_meters(meters)
        meters.register_meter("weight", AverageValueMeter())
        return meters

    def __call__(self, input1: Tensor, input2: Tensor, cur_epoch: int, **kwargs):
        assert simplex(input1)
        if self.weight == 0:
            if self.meters:
                self.meters[self.name].add(0)
            return torch.tensor(0, device=input1.device, dtype=input1.dtype)
        loss = self.criterion(self.flatten_predict(input1))

        if self.meters:
            self.criterion: IMSATDynamicWeight
            self.meters[self.name].add(loss.item())
            self.meters["weight"].add(self.criterion.dynamic_weight)

        if kwargs.get("save_image_condition", False):
            self.criterion: IMSATLoss
            joint_2D_figure(self.criterion.get_joint_matrix(), tb_writer=self.get_tb_writer(), cur_epoch=cur_epoch,
                            tag=f"{class_name(self)}_{self.name}")

        return loss * self.weight

    @staticmethod
    def flatten_predict(prediction: Tensor):
        assert prediction.dim() == 4
        b, c, h, w = prediction.shape
        prediction = torch.swapaxes(prediction, 0, 1)
        prediction = prediction.reshape(c, -1)
        prediction = torch.swapaxes(prediction, 0, 1)
        return prediction


class _ConsistencyHook(_TinyHook):
    def __init__(self, *, name: str = "consistency", weight: float) -> None:
        criterion = KL_div()
        super().__init__(name=name, criterion=criterion, weight=weight)

    def __call__(self, input1: Tensor, input2: Tensor, cur_epoch: int, **kwargs):
        assert simplex(input1)
        if self.weight == 0:
            if self.meters:
                self.meters[self.name].add(0)
            return torch.tensor(0, device=input1.device, dtype=input1.dtype)
        loss = self.criterion(input1, input2.detach())
        if self.meters:
            self.meters[self.name].add(loss.item())

        return loss * self.weight
