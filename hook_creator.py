from contrastyou.utils import class_name
from semi_seg.hooks import create_infonce_hooks, create_sp_infonce_hooks, create_discrete_mi_consistency_hook, \
    create_mt_hook, create_differentiable_mt_hook, create_ent_min_hook, create_orthogonal_hook, create_iid_seg_hook, \
    create_pseudo_label_hook, create_imsat_hook, create_consistency_hook, create_cross_correlation_hooks2, \
    create_intermediate_imsat_hook, create_uamt_hook, create_mixup_hook, create_ict_hook, create_dae_hook, \
    create_superpixel_hooks


def create_hook_from_config(model, config, *, is_pretrain=False, trainer):
    data_name = config["Data"]["name"]
    max_epoch = config["Trainer"]["max_epoch"]
    hooks = []
    if "InfonceParams" in config:
        hook = create_infonce_hooks(model=model, data_name=data_name, **config["InfonceParams"])
        hooks.append(hook)
    if "SPInfonceParams" in config:
        info_hook = create_sp_infonce_hooks(
            model=model, data_name=data_name, max_epoch=max_epoch, **config["SPInfonceParams"]
        )
        hooks.append(info_hook)
    if "DiscreteMIConsistencyParams" in config:
        if is_pretrain:
            raise RuntimeError("DiscreteMIConsistencyParams are not supported for pretrain stage")
        mi_hook = create_discrete_mi_consistency_hook(model=model, **config["DiscreteMIConsistencyParams"])
        hooks.append(mi_hook)
    if "MeanTeacherParameters" in config:
        if is_pretrain:
            raise RuntimeError("`MeanTeacherParameters` are not supported for pretrain stage")
        mt_hook = create_mt_hook(model=model, **config["MeanTeacherParameters"])
        hooks.append(mt_hook)
        trainer.set_model4inference(mt_hook.teacher_model)

    if "UAMeanTeacherParameters" in config:
        if is_pretrain:
            raise RuntimeError("`UAMeanTeacherParameters` are not supported for pretrain stage")
        uamt_hook = create_uamt_hook(model=model, **config["UAMeanTeacherParameters"])
        hooks.append(uamt_hook)
        trainer.set_model4inference(uamt_hook.teacher_model)

    if "ICTMeanTeacherParameters" in config:
        if is_pretrain:
            raise RuntimeError("`ICTMeanTeacherParameters` are not supported for pretrain stage")
        ict_hook = create_ict_hook(model=model, **config["ICTMeanTeacherParameters"])
        hooks.append(ict_hook)
        trainer.set_model4inference(ict_hook.teacher_model)

    if "DifferentiableMeanTeacherParameters" in config:
        if is_pretrain:
            raise RuntimeError("`DifferentiableMeanTeacherParameters` are not supported for pretrain stage")
        mt_hook = create_differentiable_mt_hook(model=model, **config["DifferentiableMeanTeacherParameters"])
        hooks.append(mt_hook)
        trainer.set_model4inference(mt_hook.teacher_model)
    if "EntropyMinParameters" in config:
        ent_hook = create_ent_min_hook(weight=float(config["EntropyMinParameters"]["weight"]))
        hooks.append(ent_hook)

    if "OrthogonalParameters" in config:
        orth_hook = create_orthogonal_hook(weight=config["OrthogonalParameters"]["weight"], model=model)
        hooks.append(orth_hook)

    if "IIDSegParameters" in config:
        iid_hook = create_iid_seg_hook(weight=config["IIDSegParameters"]["weight"],
                                       mi_lambda=config["IIDSegParameters"]["mi_lambda"])
        hooks.append(iid_hook)

    if "PsuedoLabelParams" in config:
        pl_hook = create_pseudo_label_hook(weight=float(config["PsuedoLabelParams"]["weight"]))
        hooks.append(pl_hook)

    if "IMSATParameters" in config:
        im_hook = create_imsat_hook(weight=float(config["IMSATParameters"]["weight"]))
        hooks.append(im_hook)
    if "IMSATFeatureParameters" in config:
        im_hook = create_intermediate_imsat_hook(**config["IMSATFeatureParameters"], model=model)
        hooks.append(im_hook)

    if any(["CrossCorrelationParameters" in x for x in config.keys()]):
        multiple_keys = [x for x in config.keys() if "CrossCorrelationParameters" in x]
        for key in multiple_keys:
            cur_config = config[key]
            hook = create_cross_correlation_hooks2(
                model=model, feature_name=cur_config["feature_name"],
                num_clusters=cur_config["num_clusters"],
                head_type=cur_config["head_type"], num_subheads=cur_config["num_subheads"],
                save=cur_config["save"], hook_params=cur_config["hooks"]
            )
            hooks.append(hook)

    if "ConsistencyParameters" in config:
        hook = create_consistency_hook(weight=config["ConsistencyParameters"]["weight"])
        hooks.append(hook)

    if "MixUpParams" in config:
        from semi_seg.trainers.trainer import MixUpTrainer
        if not isinstance(trainer, MixUpTrainer):
            raise RuntimeError(f"MixUpHook only support MixupTrainer. given {class_name(trainer)}")

        hook = create_mixup_hook(weight=float(config["MixUpParams"]["weight"]),
                                 enable_bn=config["MixUpParams"]["enable_bn"])
        hooks.append(hook)

    if "DAEParameters" in config:
        hook = create_dae_hook(weight=config["DAEParameters"]["weight"], num_classes=config["OPT"]["num_classes"])
        hooks.append(hook)

    if "InfonceSuperPixelParams" in config:
        hook = create_superpixel_hooks(
            model=model,
            weights=config["InfonceSuperPixelParams"]["weights"],
            spatial_size=config["InfonceSuperPixelParams"]["spatial_size"],
            data_name=data_name,
            feature_names=config["InfonceSuperPixelParams"]["feature_names"]
        )
        hooks.append(hook)

    return hooks
