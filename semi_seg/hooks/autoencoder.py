import torch
import typing as t
from contrastyou.hooks import TrainerHook, EpocherHook
from torch import nn, Tensor
import torch.nn.functional as F
from contrastyou.meters import AverageValueMeter

if t.TYPE_CHECKING:
    from contrastyou.meters import MeterInterface


class AuxiliaryLayer(nn.Module):
    """
    Auxiliary layer to convert a logit output to tranh or sigmoid output.
    """

    def __init__(self, *, in_features: int, out_features: int = 1, activation: str = 'sigmoid'):
        assert activation in {'sigmoid', 'tanh'}, activation

        super(AuxiliaryLayer, self).__init__()
        self.fc = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.activation = torch.nn.Sigmoid() if activation == 'sigmoid' else torch.nn.Tanh()

    def forward(self, x):
        return self.activation(self.fc(x))


class DenosingAutoEncoderTrainerHook(TrainerHook):
    def __init__(self, *, hook_name: str = "deAE", num_classes: int, weight: float = 0.0, **kwargs):
        super().__init__(hook_name=hook_name)
        self.num_classes = num_classes
        self.aux_layer = AuxiliaryLayer(in_features=num_classes, out_features=1)
        self.weight = weight

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return [self.aux_layer, ]

    def __call__(self, *args, **kwargs):
        return DenosingAutoEncoderEpocherHook(extra_layer=self.aux_layer, weight=self.weight)


class DenosingAutoEncoderEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "deAE", extra_layer: "AuxiliaryLayer", weight) -> None:
        super().__init__(name=name)
        self.layer = extra_layer
        self._weight = weight

    def _call_implementation(self, *, unlabeled_image_tf: Tensor, unlabeled_tf_logits: Tensor, **kwargs):
        recovered_output = self.layer(unlabeled_tf_logits)
        loss = F.mse_loss(recovered_output, unlabeled_image_tf)
        if self.meters:
            self.meters[self.name].add(loss.item())
        return self._weight * loss

    def configure_meters_given_epocher(self, meters: 'MeterInterface'):
        self.meters.register_meter(self.name, AverageValueMeter())