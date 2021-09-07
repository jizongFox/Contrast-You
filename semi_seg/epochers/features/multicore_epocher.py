import torch
from abc import ABC
from contrastyou.configure.manager import get_config
from contrastyou.losses.multicore_loss import MultiCoreKL
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import class2one_hot
from functools import partial
from ..epocher import EvalEpocher, SemiSupervisedEpocher


class MultiCoreTrainEpocher(SemiSupervisedEpocher, ABC):

    @property
    def num_classes(self):
        config = get_config(scope="base")
        return config["Arch"]["true_num_classes"]

    def _batch_update(self, *, cur_batch_num: int, labeled_image, labeled_target, labeled_filename, label_group,
                      unlabeled_image, unlabeled_image_tf, seed, unl_group, unl_partition, unlabeled_filename,
                      retain_graph=False, **kwargs):
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)

        with self.autocast:
            label_logits, unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
                labeled_image=labeled_image,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            unlabeled_logits_tf = self.transform_with_seed(unlabeled_logits, seed=seed)

            # supervised part
            one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            self._sup_criterion: MultiCoreKL
            sup_loss = self._sup_criterion(label_logits.softmax(1), one_hot_target)
            # regularized part
            reg_loss = self.regularization(
                seed=seed,
                labeled_image=labeled_image,
                labeled_target=labeled_target,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                unlabeled_tf_logits=unlabeled_tf_logits,
                unlabeled_logits_tf=unlabeled_logits_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                labeled_filename=labeled_filename,
                unlabeled_filename=unlabeled_filename,
                affine_transformer=partial(self.transform_with_seed, seed=seed)
            )

        total_loss = sup_loss + reg_loss
        # gradient backpropagation
        self.scale_loss(total_loss).backward(retain_graph=retain_graph)
        self.optimizer_step(self._optimizer, cur_iter=cur_batch_num)

        # recording can be here or in the regularization method
        if self.on_master():
            with torch.no_grad():
                reduced_simplex = self._sup_criterion.reduced_simplex(label_logits.softmax(1))
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(reduced_simplex.max(1)[1], labeled_target.squeeze(1),
                                            group_name=label_group)
                self.meters["reg_loss"].add(reg_loss.item())


class MultiCoreEvalEpocher(EvalEpocher):
    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(MultiCoreEvalEpocher, self).configure_meters(meters)
        meters.register_meter("true_loss", AverageValueMeter())
        return meters

    def _batch_update(self, *, eval_img, eval_target, eval_group):
        self._sup_criterion: MultiCoreKL
        with self.autocast:
            eval_logits = self._model(eval_img)
            onehot_target = class2one_hot(eval_target.squeeze(1), self.num_classes)
            eval_loss = self._sup_criterion(eval_logits.softmax(1), onehot_target, )

            reduced_simplex = self._sup_criterion.reduced_simplex(eval_logits.softmax(1))

            true_eval_loss = self._sup_criterion._kl(reduced_simplex, onehot_target)

        self.meters["loss"].add(eval_loss.item())
        self.meters["true_loss"].add(true_eval_loss.item())
        self.meters["dice"].add(reduced_simplex.max(1)[1], eval_target.squeeze(1), group_name=eval_group)

    @property
    def num_classes(self):
        config = get_config(scope="base")
        return config["Arch"]["true_num_classes"]
