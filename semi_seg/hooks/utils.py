from functools import lru_cache

from deepclustering2.configparser._utils import get_config
from loguru import logger

from semi_seg.epochers._helper import PartitionLabelGenerator, PatientLabelGenerator, ACDCCycleGenerator, \
    SIMCLRGenerator


@lru_cache()
def global_label_generator(dataset_name: str, contrast_on: str):
    if dataset_name == "acdc":
        logger.debug("initialize {} label generator for encoder training", contrast_on)
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "cycle":
            return ACDCCycleGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    elif dataset_name in ("prostate", "prostate_md"):
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    elif dataset_name == "mmwhs":
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    else:
        NotImplementedError(dataset_name)


def get_label(contrast_on, partition_group, label_group):
    dataset_name = get_config(scope="base")["Data"]["name"]
    if dataset_name == "acdc":
        labels = global_label_generator(dataset_name="acdc", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=[p.split("_")[0] for p in label_group],
             experiment_list=[p.split("_")[1] for p in label_group])
    elif dataset_name == "prostate":
        labels = global_label_generator(dataset_name="prostate", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=[p.split("_")[0] for p in label_group])
    elif dataset_name in ("mmwhsct", "mmwhsmr"):
        labels = global_label_generator(dataset_name="mmwhs", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=label_group)
    elif dataset_name == "prostate_md":
        labels = global_label_generator(dataset_name="prostate", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=label_group)
    else:
        raise NotImplementedError()
    return labels
