import os
import pprint
from functools import reduce
from typing import Optional

from loguru import logger
from torch import nn

from contrastyou.arch import UNet, UNetFeatureMapEnum
from contrastyou.arch.unet import _complete_arch_start2end
from contrastyou.configure import dictionary_merge_by_hierachy, extract_dictionary_from_anchor, \
    extract_params_with_key_prefix, ConfigManager


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


def randomnize_model(model: UNet, *, start: UNetFeatureMapEnum, end: UNetFeatureMapEnum, include_start: bool = True,
                     include_end: bool = True):
    all_component = _complete_arch_start2end(start.value, end.value, include_start=include_start,
                                             include_end=include_end)

    def initialize_weights(model):
        # Initializes weights according to the DCGAN paper
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    if len(all_component) > 0:
        logger.opt(depth=2).trace("set random initialization from {} to {}", all_component[0], all_component[-1])
    for c in all_component:
        cur_module = getattr(model, "_" + c)
        cur_module.apply(initialize_weights)
    return model
