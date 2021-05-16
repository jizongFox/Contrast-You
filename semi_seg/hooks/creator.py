from contrastyou.hooks.base import CombineTrainerHook
from .consistency import ConsistencyTrainerHook
from .discretemi import DiscreteMITrainHook
from .infonce import SelfPacedINFONCEHook


def create_iic_hook(feature_names, iic_weights, consistency_weight, model):
    mi_hooks = [DiscreteMITrainHook(name=f"iic/{f.lower()}", model=model, feature_name=f, weight=w) for f, w in
                zip(feature_names, iic_weights)]
    consistency_hook = ConsistencyTrainerHook(name="iic/consist.", weight=consistency_weight)
    return CombineTrainerHook(*mi_hooks, consistency_hook)


def create_infonce_hook(feature_names, weights, contrast_ons, mode, data_name, begin_values, end_values, max_epoch,
                        model, p=0.5):
    infonce_hooks = [
        SelfPacedINFONCEHook(name=f"infoce/{c}/{f}", model=model, feature_name=f, weight=w, data_name=data_name,
                             contrast_on=c, mode=mode, p=p, begin_value=b, end_value=e, max_epoch=max_epoch) for
        f, w, c, b, e in zip(feature_names, weights, contrast_ons, begin_values, end_values)]
    infonce_hook = CombineTrainerHook(*infonce_hooks)
    return infonce_hook
