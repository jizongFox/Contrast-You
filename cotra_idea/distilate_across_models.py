import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose

from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.meters2 import AverageValueMeter, ConfusionMatrix
from deepclustering2.models.ema import ema_updater
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.writer import SummaryWriter

train_transform = Compose([transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
val_transform = Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

train_data = CIFAR10(root="./data", download=True, transform=train_transform)
val_data = CIFAR10("./data", train=False, download=True, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=128, num_workers=8, sampler=InfiniteRandomSampler(train_data, True))
val_loader = DataLoader(val_data, batch_size=100)


class NetArray(list):
    def forward(self, input):
        return [x.forward(input) for x in self]

    def train(self):
        for i in self:
            i.train()

    def to(self, device):
        for i in self:
            i.to(device)

    def __call__(self, *args, **kwargs):
        return [i.__call__(*args, **kwargs) for i in self]


def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


from models import ResNet18 as resnet18

device = torch.device("cuda")
nets = NetArray([resnet18(num_classes=10) for _ in range(1)])
optimizer = torch.optim.SGD(itertools.chain(*(x.parameters() for x in nets)), lr=1e-1, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 100, 150], gamma=0.1)
teacher_net = resnet18(num_classes=10)

for param in teacher_net.parameters():
    param.detach_()
teacher_net.train()


class Trainer:
    def __init__(self, nets, teacher_net, optimizer, scheduler, train_loader, val_loader) -> None:
        self._nets = nets
        self._teacher_net = teacher_net
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._train_iter = iter(train_loader)
        self._val_loader = val_loader
        self._num_batches = 391
        self._sup_loss = nn.CrossEntropyLoss()
        self._ema_updater = ema_updater(justify_alpha=False, alpha=0.999, weight_decay=1e-6, update_bn=True)
        self.to(device)
        self._writer = SummaryWriter(log_dir="./main_self")

    def __len__(self):
        return len(self._nets)

    def train_epoch(self):
        self._nets.train()
        self._teacher_net.train()

        loss_meters = [AverageValueMeter() for _ in range(len(self))]
        acc_meters = [ConfusionMatrix(10) for _ in range(len(self))]
        indicator = tqdm(range(self._num_batches))
        indicator.set_description(f"training epoch {self._epoch}")

        for i, (image, target) in zip(indicator, self._train_iter):
            image, target = image.to(device), target.to(device)
            logit_list = nets(image)
            loss_list = list(map(lambda x: self._sup_loss(x, target), logit_list))
            loss = average_iter(loss_list)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            list(map(lambda meter, loss: meter.add(loss.item()), loss_meters, loss_list))
            list(map(lambda meter, logits: meter.add(logits.max(1)[1], target), acc_meters, logit_list))
            report_dict = {"training loss": average_iter([x.summary()["mean"] for x in loss_meters]),
                           "training acc": average_iter([x.summary()["acc"] for x in acc_meters])}
            for n in self._nets:
                self._ema_updater(self._teacher_net, n)
            indicator.set_postfix(report_dict)

        print("training loss:", [x.summary()["mean"] for x in loss_meters])
        print("training acc:", [x.summary()["acc"] for x in acc_meters])
        self._writer.add_scalar("train/loss", average_iter([x.summary()["mean"] for x in loss_meters]),
                                global_step=self._epoch)
        self._writer.add_scalar("train/acc", average_iter([x.summary()["acc"] for x in acc_meters]),
                                global_step=self._epoch)
        self._writer.add_scalar("train/lr", get_lrs_from_optimizer(self._optimizer)[0], global_step=self._epoch)

    @torch.no_grad()
    def val_epoch(self, model):
        model.eval()
        acc_meter = ConfusionMatrix(10)
        total = 0
        correct = 0
        val_loader = tqdm(self._val_loader)
        val_loader.set_description(f"validation epoch {self._epoch}")
        for i, (image, target) in enumerate(val_loader):
            image, target = image.to(device), target.to(device)
            logit = model(image)
            acc_meter.add(logit.max(1)[1], target)
            total += image.size(0)
            correct += torch.eq(logit.max(1)[1], target).sum()
            report_dict = {"validating acc": acc_meter.summary()["acc"]}
            val_loader.set_postfix(report_dict)
        print("validating acc:", acc_meter.summary()["acc"])
        self._writer.add_scalar("val/acc", acc_meter.summary()["acc"], global_step=self._epoch)

    def train(self):
        for self._epoch in range(200):
            self.train_epoch()
            self.val_epoch(self._teacher_net)
            self._scheduler.step()

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                try:
                    v.to(device)
                except:
                    continue


trainer = Trainer(nets, teacher_net, optimizer, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader)
trainer.train()
