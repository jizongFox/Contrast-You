from scipy.sparse import issparse  # noqa

_ = issparse  # noqa
import argparse
from itertools import chain

import numpy as np
import torch
from cifar_semi_dataset import CIFAR10
from models import ResNet18 as resnet18
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from utils import ToMixin

from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.trainer._utils import ProjectionHead
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.meters2 import AverageValueMeter, ConfusionMatrix
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.models.ema import ema_updater
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.schedulers.customized_scheduler import RampScheduler
from deepclustering2.tqdm import tqdm
from deepclustering2.utils import fix_all_seed
from deepclustering2.writer import SummaryWriter

fix_all_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--num_labeled_data", type=int, default=4000)
parser.add_argument("-r", "--meanteacher_reg", type=float, required=True)
parser.add_argument("-c", "--contrast_reg", type=float, required=True)
parser.add_argument("-s", "--save_dir", type=str, required=True)
parser.add_argument("--num_batches", default=500, type=int)

args = parser.parse_args()


class TwiceTransformation:

    def __init__(self, transform) -> None:
        self._transform = transform

    def __call__(self, *args, **kwargs):
        return [self._transform(*args, **kwargs) for _ in range(2)]


train_transform = Compose([transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
val_transform = Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

CIFAR_LENGTH = 50000
LABELED_LENGTH = args.num_labeled_data
labeled_list = np.random.permutation(range(CIFAR_LENGTH))[:LABELED_LENGTH]

labeled_data = CIFAR10.create_semi_dataset(
    root="./data", download=True, transform=train_transform,
    selected_index_list=labeled_list
)
unlabeled_data = CIFAR10.create_semi_dataset(
    root="./data", download=True,
    transform=TwiceTransformation(train_transform),
)
val_data = CIFAR10("./data", train=False, download=True, transform=val_transform)
labeled_loader = DataLoader(labeled_data, batch_size=64, num_workers=8,
                            sampler=InfiniteRandomSampler(labeled_data, True))
unlabeled_loader = DataLoader(unlabeled_data, batch_size=64, num_workers=8,
                              sampler=InfiniteRandomSampler(unlabeled_data, True))
val_loader = DataLoader(val_data, batch_size=100, num_workers=2)

device = torch.device("cuda")
net = resnet18(num_classes=10)

projector_student = ProjectionHead(input_dim=512, output_dim=64, interm_dim=128, head_type="mlp")
projector_teacher = ProjectionHead(input_dim=512, output_dim=64, interm_dim=128, head_type="mlp")

optimizer = torch.optim.Adam(
    chain(net.parameters(), projector_student.parameters(), projector_teacher.parameters()),
    lr=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0)

teacher_net = resnet18(num_classes=10)
for param in teacher_net.parameters():
    param.detach_()
teacher_net.train()

rampup_scheduler = RampScheduler(begin_epoch=10, max_epoch=50, min_value=0, max_value=args.contrast_reg)


class Trainer(ToMixin):
    def __init__(self, net, teacher_net, projector_s, projector_t, optimizer, scheduler, labeled_loader,
                 unlabeled_loader, val_loader, contrastive_weight_scheduler: RampScheduler,
                 save_dir: str, num_batches=300, reg_weight: float = 10, max_epoch=100,
                 use_estimated_info=True) -> None:
        self._net = net
        self._teacher_net = teacher_net
        self._projector_s = projector_s
        self._projector_t = projector_t
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._labeled_iter = iter(labeled_loader)
        self._unlabeled_iter = iter(unlabeled_loader)
        self._val_loader = val_loader
        self._num_batches = num_batches
        self._sup_criterion = nn.CrossEntropyLoss()
        self._reg_criterion = nn.MSELoss()
        self._contrastive_criterion = SupConLoss()
        self._ema_updater = ema_updater(justify_alpha=True, alpha=0.999, weight_decay=0, update_bn=True)
        self._writer = SummaryWriter(log_dir=save_dir)
        self._contrastive_reg_scheduler = contrastive_weight_scheduler
        self._reg_weight = reg_weight
        self._max_epoch = max_epoch
        self._use_estimate_info = use_estimated_info
        self.to(device)

    def train_epoch(self):
        self._net.train()
        self._teacher_net.train()
        total_loss_meter = AverageValueMeter()
        sup_loss_meter = AverageValueMeter()
        reg_loss_meter = AverageValueMeter()
        contrast_loss_meter = AverageValueMeter()
        labeled_acc_meter = ConfusionMatrix(10)
        with tqdm(range(self._num_batches)) as indicator:
            indicator.set_description(f"training epoch {self._epoch}")
            report_dict = {
                "lr": get_lrs_from_optimizer(self._optimizer)[0],
                "mt_weight": self._reg_weight,
                "ct_weight": self._contrastive_reg_scheduler.value
            }
            for i, (image, target), \
                ((uimage, uimage_tf), _) in zip(indicator, self._labeled_iter, self._unlabeled_iter):
                image, target = image.to(device), target.to(device)
                uimage, uimage_tf = uimage.to(device), uimage_tf.to(device)
                student_logits, student_features = self._net(torch.cat([image, uimage], dim=0))
                labeled_logits, stduent_unlabeled_logits = torch.split(student_logits, [len(image), len(uimage)], dim=0)
                _, student_unlabeled_features = torch.split(student_features, [len(image), len(uimage)], dim=0)
                sup_loss = self._sup_criterion(labeled_logits, target)
                with torch.no_grad():
                    teacher_logits, teacher_unlabeled_features = self._net(uimage_tf)
                assert student_unlabeled_features.shape == teacher_unlabeled_features.shape, \
                    (student_unlabeled_features.shape, teacher_unlabeled_features.shape)
                assert teacher_logits.shape == stduent_unlabeled_logits.shape, \
                    (teacher_logits.shape, stduent_unlabeled_logits.shape)
                reg_loss = self._reg_criterion(stduent_unlabeled_logits.softmax(1), teacher_logits.softmax(1).detach())
                student_vectors = F.normalize(self._projector_s(student_unlabeled_features), dim=1)
                teacher_vectors = F.normalize(self._projector_t(teacher_unlabeled_features), dim=1)

                contrastive_loss = self._contrastive_criterion(
                    torch.stack([student_vectors, teacher_vectors], dim=1),
                    labels=teacher_logits.max(1)[1].tolist() if self._use_estimate_info else None
                )
                total_loss = sup_loss + reg_loss + self._reg_weight * contrastive_loss + self._contrastive_reg_scheduler.value * contrastive_loss
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                with torch.no_grad():
                    total_loss_meter.add(total_loss.item())
                    sup_loss_meter.add(sup_loss.item())
                    reg_loss_meter.add(reg_loss.item())
                    contrast_loss_meter.add(contrastive_loss.item())
                    labeled_acc_meter.add(labeled_logits.max(1)[1], target)
                    report_dict.update({
                        "ttlloss": total_loss_meter.summary()["mean"],
                        "suploss": sup_loss_meter.summary()["mean"],
                        "regloss": reg_loss_meter.summary()["mean"],
                        "conloss": contrast_loss_meter.summary()["mean"],
                        "labacc": labeled_acc_meter.summary()["acc"]
                    })
                    self._ema_updater(self._teacher_net, self._net)
                    indicator.set_postfix(report_dict)
        return report_dict

    @torch.no_grad()
    def val_epoch(self, model):
        model.eval()
        acc_meter = ConfusionMatrix(10)
        with tqdm(self._val_loader) as val_loader:
            val_loader.set_description(f"validation epoch {self._epoch}")
            report_dict = {}
            for i, (image, target) in enumerate(val_loader):
                image, target = image.to(device), target.to(device)
                logit, _ = model(image)
                acc_meter.add(logit.max(1)[1], target)
                report_dict = {"validating acc": acc_meter.summary()["acc"]}
                val_loader.set_postfix(report_dict)
        return report_dict

    def train(self):
        for self._epoch in range(self._max_epoch):
            train_dict = self.train_epoch()
            val_student_dict = self.val_epoch(self._net)
            val_teacher_dict = self.val_epoch(self._teacher_net)
            self._scheduler.step()
            self._contrastive_reg_scheduler.step()
            result = StorageIncomeDict(train=train_dict, val_student=val_student_dict, val_teacher=val_teacher_dict)
            self._writer.add_scalar_with_StorageDict(result, self._epoch)


trainer = Trainer(net, teacher_net, projector_student, projector_teacher, optimizer, scheduler=scheduler,
                  labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader, val_loader=val_loader,
                  contrastive_weight_scheduler=rampup_scheduler,
                  save_dir=args.save_dir, reg_weight=args.meanteacher_reg, num_batches=args.num_batches)
trainer.train()
