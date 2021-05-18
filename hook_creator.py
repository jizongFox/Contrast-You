from semi_seg.hooks import create_infonce_hooks


def create_hook_from_config(model, config, is_pretrain=False):
    # iic_hook = create_discrete_mi_consistency_hook(model=model, feature_names=["Conv5", "Up_conv3"],
    #                                                mi_weights=[0.1, 0.05], dense_paddings=[1],
    #                                                consistency_weight=1.0)
    # info_hook = create_sp_infonce_hooks(model=model, feature_names=["Conv5", ], weights=0.1,
    #                                     contrast_ons=["partition", ], data_name="acdc", begin_values=1e6,
    #                                     end_values=1e6, mode="soft", max_epoch=10, correct_grad=True)
    info_hook = create_infonce_hooks(model=model, feature_names=["Conv5", ], weights=1,
                                     contrast_ons=["partition", ], data_name="acdc")
    return (info_hook,)
