import os

from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div

from contrastyou import CONFIG_PATH
from semi_seg.arch import UNet
from semi_seg.data import get_data_loaders, create_val_loader
from semi_seg.mi_estimator.discrete_mi_estimator2 import MIEstimatorHook
from semi_seg.trainers.zz_experiment import SemiTrainer2

model = UNet(num_classes=4, input_dim=1)


def get_data_params():
    return {"name": "acdc",
            "labeled_data_ratio": 0.1}


def get_loader_params():
    return {"shuffle": True, "num_workers": 8, "batch_size": 2}


labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
    data_params=get_data_params(), labeled_loader_params=get_loader_params(),
    unlabeled_loader_params=get_loader_params(),
)
val_loader, test_loader = create_val_loader(test_loader=test_loader)
with ConfigManger(base_path=os.path.join(CONFIG_PATH, "base.yaml"))(scope="base") as config:
    trainer = SemiTrainer2(model=model, labeled_loader=labeled_loader,
                           unlabeled_loader=unlabeled_loader, val_loader=val_loader, test_loader=test_loader,
                           sup_criterion=KL_div(), save_dir="base", max_epoch=100, num_batches=100, device="cuda",
                           configuration=config)
    trainer.add_hook(MIEstimatorHook(model=model, feature_name="Conv5", weight=0.1, num_clusters=20, num_subheads=5))
    trainer.add_hook(MIEstimatorHook(model=model, feature_name="Conv4", weight=0.1, num_clusters=20, num_subheads=5))

    trainer.init()

    trainer.start_training()
