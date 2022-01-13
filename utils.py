import contextlib
import os
import pprint
import typing
from functools import reduce
from typing import Optional

from loguru import logger
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from contrastyou.configure import dictionary_merge_by_hierachy, extract_dictionary_from_anchor, \
    extract_params_with_key_prefix, ConfigManager
from contrastyou.data import DatasetBase
from contrastyou.utils import get_dataset
from demo.model import _TwinBNCONTEXT


def separate_pretrain_finetune_configs(config_manager):
    input_params = config_manager.parsed_config
    base_config = config_manager.base_config
    opt_params = reduce(dictionary_merge_by_hierachy, config_manager.optional_configs)

    pretrain_config = dictionary_merge_by_hierachy(base_config, opt_params)

    # extract the input_params for both settings
    pretrain_config = dictionary_merge_by_hierachy(
        pretrain_config,
        extract_dictionary_from_anchor(target_dictionary=input_params,
                                       anchor_dictionary=pretrain_config,
                                       prune_anchor=True))
    # extract input_params for pre_
    pretrain_config = dictionary_merge_by_hierachy(
        pretrain_config,
        extract_params_with_key_prefix(input_params, prefix="pre_"))

    base_config = dictionary_merge_by_hierachy(
        base_config,
        extract_dictionary_from_anchor(target_dictionary=input_params,
                                       anchor_dictionary=base_config,
                                       prune_anchor=True))
    base_config = dictionary_merge_by_hierachy(
        base_config,
        extract_params_with_key_prefix(input_params, prefix="ft_"))

    return pretrain_config, base_config


def logging_configs(manager: ConfigManager, logger: logger):
    unmerged_dictionaries = manager.unmerged_configs
    parsed_params = manager.parsed_config
    config_dictionary = manager.config
    for i, od in enumerate(unmerged_dictionaries):
        logger.info(f"optional configs {i}")
        logger.info("\n" + pprint.pformat(od))
    logger.info(f"parsed params")
    logger.info("\n" + pprint.pformat(parsed_params))
    logger.info("merged params")
    logger.info("\n" + pprint.pformat(config_dictionary))


def find_checkpoint(trainer_folder, name="last.pth") -> Optional[str]:
    ckpt_path = os.path.join(trainer_folder, name)
    if os.path.exists(ckpt_path):
        logger.info(f"Found existing checkpoint from folder {trainer_folder}")
        return ckpt_path
    return None


def grouper(array_list, group_num):
    num_samples = len(array_list) // group_num
    batch = []
    for item in array_list:
        if len(batch) == num_samples:
            yield batch
            batch = []
        batch.append(item)
    if len(batch) > 0:
        yield batch


class ColorInverseTransform:
    def __call__(self, image):
        return 1 - image


def _make_da_dataloader(dataloader: DataLoader):
    dataset: 'DatasetBase' = get_dataset(dataloader)
    previous_img_transform = dataset._transforms._image_transform
    dataset._transforms._image_transform = Compose([previous_img_transform, ColorInverseTransform()])
    return dataloader


def make_data_dataloaders(*dataloader: DataLoader) -> typing.Tuple[DataLoader, ...]:
    return tuple([_make_da_dataloader(loader) for loader in dataloader])


class TwinBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super(TwinBatchNorm2d, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.bn2 = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self._indicator = 0

    def forward(self, x):
        if len(_TwinBNCONTEXT) == 0:
            raise RuntimeError(f"{self.__class__.__name__} must be used with context manager of `switch_bn`.")
        if self.indicator == 0:
            return self.bn1(x)
        else:
            return self.bn2(x)

    @property
    def indicator(self):
        return self._indicator

    @indicator.setter
    def indicator(self, value):
        assert value in (0, 1)
        self._indicator = value


def convert2TwinBN(module: nn.Module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = TwinBatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            device=module.running_mean.device,
            dtype=module.running_mean.dtype,
        )
    for name, child in module.named_children():
        module_output.add_module(
            name, convert2TwinBN(child)
        )
    del module
    return module_output


@contextlib.contextmanager
def switch_bn(module: nn.Module, indicator: int):
    _TwinBNCONTEXT.append("A")
    previous_state = {n: v.indicator for n, v in module.named_modules() if isinstance(v, TwinBatchNorm2d)}
    for n, v in module.named_modules():
        if isinstance(v, TwinBatchNorm2d):
            v.indicator = indicator
    yield
    for n, v in module.named_modules():
        if isinstance(v, TwinBatchNorm2d):
            v.indicator = previous_state[n]
    _TwinBNCONTEXT.pop()
