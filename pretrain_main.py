import os
from copy import deepcopy as dcopy
from functools import reduce
from typing import Dict, Any

from deepclustering2.loss import KL_div
from loguru import logger

from contrastyou import CONFIG_PATH, success
from contrastyou.config import ConfigManger, dictionary_merge_by_hierachy, extract_dictionary_from_anchor
from contrastyou.mytqdm.utils import is_iterable
from contrastyou.types import is_map
from contrastyou.utils import fix_all_seed_within_context, config_logger, set_deterministic, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg import ratio_zoo
from semi_seg.arch import UNet
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers.new_pretrain import PretrainTrainer
from val import val


def extract_params_with_key_prefix(dictionary: Dict[str, Any], prefix: str) -> Dict:
    result_dict = {}
    for k, v in dictionary.items():
        if is_map(v):
            result_dict[k] = extract_params_with_key_prefix(v, prefix=prefix)
        elif is_iterable(v):
            result_dict[k] = [extract_params_with_key_prefix(x, prefix=prefix) for x in v]
        else:
            if k.startswith(prefix):
                result_dict[k.replace(prefix, "")] = v

        # clean items with {}
        for _k, _v in result_dict.copy().items():
            if _v == {}:
                del result_dict[_k]
    return result_dict


def main():
    config_manager = ConfigManger(
        base_path=os.path.join(CONFIG_PATH, "base.yaml"),
        optional_paths=os.path.join(CONFIG_PATH, "pretrain.yaml"), strict=False, verbose=False
    )
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

    with config_manager(scope="base") as config:
        seed = config.get("RandomSeed", 10)
        data_name = config["Data"]["name"]
        absolute_save_dir = os.path.abspath(os.path.join(PretrainTrainer.RUN_PATH, str(config["Trainer"]["save_dir"])))
        config_logger(absolute_save_dir)
        with fix_all_seed_within_context(seed):
            model = worker(pretrain_config, absolute_save_dir, seed)

        val(model=model, save_dir=absolute_save_dir, base_config=base_config, seed=seed,
            labeled_ratios=ratio_zoo[data_name])


def worker(config, absolute_save_dir, seed, ):
    config = dcopy(config)
    model_checkpoint = config["Arch"].pop("checkpoint", None)
    with fix_all_seed_within_context(seed):
        model = UNet(**config["Arch"])
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=True)
    #
    trainer = PretrainTrainer(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                              val_loader=val_loader, test_loader=test_loader,
                              criterion=KL_div(verbose=False), config=config,
                              save_dir=os.path.join(absolute_save_dir, "pre"),
                              **{k: v for k, v in config["Trainer"].items() if k != "save_dir"})
    with fix_all_seed_within_context(seed):
        hooks = create_hook_from_config(model, config, is_pretrain=True)
    trainer.register_hooks(*hooks)
    until = feature_until_from_hooks(*hooks)
    trainer.forward_until = until
    with model.set_grad(False, start=until, include_start=False):
        trainer.init()
        trainer.start_training()
    success(save_dir=trainer.save_dir)
    return model


if __name__ == '__main__':
    set_deterministic(True)
    main()
