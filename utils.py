import os
from functools import reduce
from typing import Optional

from loguru import logger
from omegaconf import OmegaConf
from prettytable.colortable import ColorTable, Themes

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
    unmerged_dictionaries = manager.base_config
    parsed_params = manager.cmd_config
    config_dictionary = manager.config
    logger.info("base params")
    # logger.info("\n" + OmegaConf.to_yaml(unmerged_dictionaries))
    logger.info("parsed params")
    # logger.info("\n" + OmegaConf.to_yaml(parsed_params))
    logger.info("merged params")
    # logger.info("\n" + OmegaConf.to_yaml(config_dictionary))
    x = ColorTable(theme=Themes.OCEAN)
    x.add_column("base params",
                 [OmegaConf.to_yaml(unmerged_dictionaries)], align="l")
    x.add_column("parsed params", [OmegaConf.to_yaml(parsed_params), ], align="l", valign="t")
    x.add_column("merged params",
                 [OmegaConf.to_yaml(config_dictionary), ], align="l")
    logger.info("\n" + str(x))


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
