import os
from functools import reduce
from typing import Optional

from loguru import logger

from contrastyou.configure import dictionary_merge_by_hierachy, extract_dictionary_from_anchor, \
    extract_params_with_key_prefix, ConfigManger


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


def logging_configs(manager: ConfigManger, logger):
    import pprint
    optional_dictionaries = manager.optional_configs
    parsed_params = manager.parsed_config
    config_dictionary = manager.config
    for i, od in enumerate(optional_dictionaries):
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
