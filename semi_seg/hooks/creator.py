from typing import List, Union, TypeVar, Sequence

from torch import nn

from contrastyou.arch import UNet
from contrastyou.hooks.base import CombineTrainerHook, TrainerHook
from contrastyou.utils.utils import ntuple, class_name
from .ccblock import CrossCorrelationHook
from .consistency import ConsistencyTrainerHook
from .discretemi import DiscreteMITrainHook
from .dmt import DifferentiableMeanTeacherTrainerHook
from .entmin import EntropyMinTrainerHook
from .infonce import SelfPacedINFONCEHook, INFONCEHook
from .midl import IIDSegmentationTrainerHook
from .midl import IMSATTrainHook
from .mt import MeanTeacherTrainerHook
from .orthogonal import OrthogonalTrainerHook
from .pseudolabel import PseudoLabelTrainerHook

decoder_names = UNet.decoder_names
encoder_names = UNet.encoder_names

T = TypeVar("T")
item_or_seq = Union[T, Sequence[T]]


def get_individual_hook(*hooks):
    for h in hooks:
        assert isinstance(h, TrainerHook)
        if isinstance(h, CombineTrainerHook):
            yield from get_individual_hook(*h._hooks)  # noqa
        else:
            yield h


def mt_in_hooks(*hooks) -> bool:
    for h in get_individual_hook(hooks):
        if isinstance(h, MeanTeacherTrainerHook):
            return True
    return False


def feature_until_from_hooks(*hooks) -> Union[str, None]:
    hook_iter = get_individual_hook(*hooks)
    feature_name_list = [h._feature_name for h in hook_iter if hasattr(h, "_feature_name")]
    if len(feature_name_list) > 0:
        from contrastyou.arch import sort_arch
        return sort_arch(feature_name_list)[-1]
    return UNet.arch_elements[-1]


def create_consistency_hook(weight: float):
    return ConsistencyTrainerHook(name="consistency", weight=weight)


def create_discrete_mi_hooks(*, feature_names: List[str], weights: List[float], paddings: List[int],
                             model: nn.Module):
    assert len(feature_names) == len(weights), (feature_names, weights)
    decoder_features = [f for f in feature_names if f in decoder_names]

    assert len(paddings) == len(decoder_features), (decoder_features, paddings)
    _pad_gen = iter(paddings)
    paddings_ = [next(_pad_gen) if f in decoder_features else None for f in feature_names]

    hooks = [DiscreteMITrainHook(name=f"discreteMI/{f.lower()}", model=model, feature_name=f, weight=w, padding=p) for
             f, w, p in zip(feature_names, weights, paddings_)]
    return CombineTrainerHook(*hooks)


def create_discrete_mi_consistency_hook(*, model: nn.Module, feature_names: Union[str, List[str]],
                                        mi_weights: Union[float, List[float]],
                                        dense_paddings: List[int] = None, consistency_weight: float):
    if isinstance(feature_names, str):
        n_features = 1
    else:
        n_features = len(feature_names)
    pair_generator = ntuple(n_features)
    feature_names = pair_generator(feature_names)
    mi_weights = pair_generator(mi_weights)
    n_dense_features = len([f for f in feature_names if f in decoder_names])
    dense_paddings = ntuple(n_dense_features)(dense_paddings)
    discrete_mi_hook = create_discrete_mi_hooks(
        feature_names=feature_names, weights=mi_weights,
        paddings=dense_paddings, model=model)
    consistency_hook = create_consistency_hook(weight=consistency_weight)
    return CombineTrainerHook(discrete_mi_hook, consistency_hook)


def _infonce_hook(*, model: nn.Module, feature_name: str, weight: float, contrast_on: str, data_name: str, ):
    return INFONCEHook(name=f"infonce/{feature_name}/{contrast_on}", model=model, feature_name=feature_name,
                       weight=weight,
                       data_name=data_name, contrast_on=contrast_on)


def _infonce_sp_hook(*, model: nn.Module, feature_name: str, weight: float, contrast_on: str, data_name: str,
                     begin_value: float = 1e6, end_value: float = 1e6, max_epoch: int, mode: str = "soft", p=0.5,
                     correct_grad=False):
    return SelfPacedINFONCEHook(name=f"spinfoce/{feature_name}/{contrast_on}", model=model, feature_name=feature_name,
                                weight=weight, data_name=data_name, contrast_on=contrast_on, mode=mode, p=p,
                                begin_value=begin_value, end_value=end_value, max_epoch=max_epoch,
                                correct_grad=correct_grad)


def create_infonce_hooks(*, model: nn.Module, feature_names: Union[str, List[str]], weights: Union[float, List[float]],
                         contrast_ons: Union[str, List[str]], data_name: str, ):
    if isinstance(feature_names, str):
        num_features = 1
    else:
        num_features = len(feature_names)
    pair_generator = ntuple(num_features)

    feature_names = pair_generator(feature_names)
    weights = pair_generator(weights)
    contrast_ons = pair_generator(contrast_ons)

    hooks = [_infonce_hook(model=model, feature_name=f, weight=w, contrast_on=c, data_name=data_name) for f, w, c in
             zip(feature_names, weights, contrast_ons)]

    return CombineTrainerHook(*hooks)


def create_sp_infonce_hooks(*, model: nn.Module, feature_names: Union[str, List[str]],
                            weights: Union[float, List[float]], contrast_ons: Union[str, List[str]], data_name: str,
                            begin_values: Union[float, List[float]] = 1e10,
                            end_values: Union[float, List[float]] = 1e10, mode: str, p=0.5, max_epoch: int,
                            correct_grad: Union[bool, List[bool]] = False):
    if isinstance(feature_names, str):
        num_features = 1
    else:
        num_features = len(feature_names)
    pair_generator = ntuple(num_features)

    feature_names = pair_generator(feature_names)
    weights = pair_generator(weights)
    contrast_ons = pair_generator(contrast_ons)
    begin_values = pair_generator(begin_values)
    end_values = pair_generator(end_values)
    correct_grad = pair_generator(correct_grad)

    hooks = [_infonce_sp_hook(model=model, feature_name=f, weight=w, contrast_on=c, data_name=data_name,
                              begin_value=b, end_value=e, max_epoch=max_epoch, mode=mode, p=p, correct_grad=g)
             for f, w, c, b, e, g in zip(feature_names, weights, contrast_ons, begin_values, end_values, correct_grad)]

    return CombineTrainerHook(*hooks)


def create_mt_hook(*, model: nn.Module, weight: float, alpha: float = 0.999, weight_decay: float = 0.000001,
                   update_bn: bool = False, num_teachers: int = 1, hard_clip: bool = False):
    hook = MeanTeacherTrainerHook(name="mt", weight=weight, model=model, alpha=alpha, weight_decay=weight_decay,
                                  update_bn=update_bn, num_teachers=num_teachers, hard_clip=hard_clip)
    return hook


def create_differentiable_mt_hook(*, model: nn.Module, weight: float, alpha: float = 0.999,
                                  weight_decay: float = 0.000001, meta_weight=0, meta_criterion: str,
                                  method_name: str, ):
    hook = DifferentiableMeanTeacherTrainerHook(name="dmt", weight=weight, model=model, alpha=alpha,
                                                weight_decay=weight_decay, meta_weight=meta_weight,
                                                meta_criterion=meta_criterion, method_name=method_name)
    return hook


def create_ent_min_hook(*, weight: float = 0.001):
    hook = EntropyMinTrainerHook(name="entropy", weight=weight)
    return hook


def create_orthogonal_hook(*, weight: float = 0.001, model: UNet):
    if isinstance(model, UNet):
        prototypes = model._Deconv_1x1.weight  # noqa
    else:
        raise NotImplementedError(class_name(model))
    hook = OrthogonalTrainerHook(hook_name="orth", prototypes=prototypes, weight=weight)
    return hook


def create_iid_seg_hook(*, weight: float = 0.001, mi_lambda=1.0):
    return IIDSegmentationTrainerHook(hook_name="iid", weight=weight, mi_lambda=mi_lambda)


def create_pseudo_label_hook(*, weight: float):
    return PseudoLabelTrainerHook(weight=weight, name="plab")


def create_imsat_hook(*, weight: float = 0.1):
    return IMSATTrainHook(weight=weight)


# def create_cross_correlation_hook(*, weight: float, kernel_size: int, device: str):
#     return CrossCorrelationHook(weight=weight, kernel_size=kernel_size, device=device)


def create_cross_correlation_hooks(
    *, model: nn.Module, feature_names: item_or_seq[str], cc_weights: item_or_seq[float],
    mi_weights: item_or_seq[float], num_clusters: item_or_seq[int], kernel_size: item_or_seq[int],
    head_type=item_or_seq[str], num_subheads: item_or_seq[int],
):
    if isinstance(feature_names, str):
        num_features = 1
    else:
        num_features = len(feature_names)
    pair_generator = ntuple(num_features)

    feature_names = pair_generator(feature_names)
    cc_weights = pair_generator(cc_weights)
    mi_weights = pair_generator(mi_weights)
    num_clusters = pair_generator(num_clusters)
    kernel_size = pair_generator(kernel_size)
    head_type = pair_generator(head_type)
    num_subheads = pair_generator(num_subheads)
    hooks = []
    for cw, mw, f_name, ksize, h_type, n_subheads, n_cluster in zip(cc_weights, mi_weights, feature_names, kernel_size,
                                                                   head_type,
                                                                   num_subheads, num_clusters):
        project_params = {"num_clusters": n_cluster,
                          "head_type": h_type,
                          "normalize": False,
                          "num_subheads": n_subheads,
                          "hidden_dim": 64}
        _hook = CrossCorrelationHook(name=f"{f_name}", cc_weight=cw, feature_name=f_name, kernel_size=ksize,
                                     projector_params=project_params, model=model, mi_weight=mw)
        hooks.append(_hook)

    return CombineTrainerHook(*hooks)
