from contrastyou.hooks.base import CombineTrainerHook
from .consistency import ConsistencyTrainerHook
from .discrete_mi import MIEstimatorHook


def create_iic_hook(feature_names, iic_weights, consistency_weight, model):
    mi_hooks = [MIEstimatorHook(name=f"iic/{f}", model=model, feature_name=f, weight=w) for f, w in
                zip(feature_names, iic_weights)]
    consistency_hook = ConsistencyTrainerHook(name="iic/Consistency", weight=consistency_weight)
    return CombineTrainerHook(*mi_hooks, consistency_hook)
