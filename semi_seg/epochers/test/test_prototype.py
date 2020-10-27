import os
from itertools import chain
from unittest import TestCase

from torch import optim

from contrastyou import CONFIG_PATH
from contrastyou.arch import UNet
from contrastyou.losses.contrast_loss import SupConLoss
from deepclustering2.loss import KL_div, OrderedDict
from deepclustering2.utils import yaml_load, set_benchmark
from semi_seg._utils import ContrastiveProjectorWrapper, ClusterProjectorWrapper, IICLossWrapper
from semi_seg.dsutils import get_dataloaders


class TestPrototype(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._config = yaml_load(os.path.join(CONFIG_PATH, "semi.yaml"))
        self._config["LabeledData"]["batch_size"] = 2
        self._config["UnlabeledData"]["batch_size"] = 2

        self._model = UNet(input_dim=1, num_classes=4)

        self._labeled_loader, self._unlabeled_loader, _ = get_dataloaders(config=self._config, group_val_patient=True)
        self._sup_criterion = KL_div()
        self._device = "cuda"
        self._feature_position = ["Conv5", ]
        self._feature_importance = [1, ]
        self._projector = ContrastiveProjectorWrapper()
        self._projector.init_encoder(feature_names=self._feature_position, )
        self._projector.init_decoder(feature_names=self._feature_position, )
        self._iic_projector = ClusterProjectorWrapper()
        self._iic_projector.init_encoder(feature_names=self._feature_position, )
        self._iic_projector.init_decoder(feature_names=self._feature_position)
        self._iic_criterion = IICLossWrapper(self._feature_position, paddings=[1], patch_sizes=2048)
        self._optimizer = optim.Adam(
            chain(self._model.parameters(), self._projector.parameters(), self._iic_projector.parameters()), lr=1e-3)

        self._buffer = OrderedDict()
        for f in self._feature_position:
            self._buffer[f] = OrderedDict()
        self._info_nce = SupConLoss()
        set_benchmark(0)

    def test_epocher(self):
        from semi_seg.epochers.miepocher import IICTrainEpocher
        epocher = IICTrainEpocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=iter(self._labeled_loader),
            unlabeled_loader=iter(self._unlabeled_loader), sup_criterion=self._sup_criterion, num_batches=100,
            cur_epoch=0, device=self._device, feature_position=self._feature_position,
            feature_importance=self._feature_importance
        )
        epocher.init(reg_weight=1.0, projectors_wrapper=self._iic_projector, feature_buffers=self._buffer,
                     infoNCE_criterion=self._info_nce, IIDSegCriterionWrapper=self._iic_criterion)
        return epocher.run()
