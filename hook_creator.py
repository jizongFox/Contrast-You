from semi_seg.hooks import create_infonce_hooks, create_sp_infonce_hooks, create_discrete_mi_consistency_hook, \
    create_mt_hook, create_differentiable_mt_hook, create_ent_min_hook, create_orthogonal_hook, create_iid_seg_hook, \
    create_pseudo_label_hook, create_imsat_hook, create_consistency_hook, create_cross_correlation_hooks


def _hook_config_validator(config, is_pretrain):
    if is_pretrain:
        """Do not accept MeanTeacher and Consistency"""
        pass


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

    if "CrossCorrelationParameters" in config:
        cur_config = config["CrossCorrelationParameters"]
        hook = create_cross_correlation_hooks(
            model=model, feature_names=cur_config["feature_names"], cc_weights=cur_config["cc_weights"],
            num_clusters=cur_config["num_clusters"], kernel_size=cur_config["kernel_size"],
            head_type=cur_config["head_type"], num_subheads=cur_config["num_subheads"],
            mi_weights=cur_config["mi_weights"], save=cur_config["save"],
            padding=cur_config["IID"]["padding"], lamda=cur_config["IID"]["lamda"],
            power=cur_config["norm"]["power"], adding_coordinates=cur_config["adding_coordinates"],
            image_diff=cur_config["image_diff"]
        )
        hooks.append(hook)

    if "ConsistencyParameters" in config:
        hook = create_consistency_hook(weight=config["ConsistencyParameters"]["weight"])
        hooks.append(hook)

    return hooks
