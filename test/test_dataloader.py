from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.dataloader._seg_datset import ContrastBatchSampler
from contrastyou.dataloader.acdc_dataset import ACDCDataset

root = DATA_PATH
from deepclustering.augment import SequentialWrapper, pil_augment
from contrastyou.augment import SequentialWrapperTwice

transform = SequentialWrapper(
    pil_augment.Compose([
        # pil_augment.RandomCrop(128),
        pil_augment.RandomRotation(40),
        pil_augment.ToTensor()
    ]),
    pil_augment.Compose([
        # pil_augment.RandomCrop(128),
        pil_augment.RandomRotation(40),

        pil_augment.ToLabel()
    ]),
    if_is_target=[False, True]
)
twicetransform = SequentialWrapperTwice(transform)

dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=transform)
print(dataset.show_group_set(), dataset.show_parition_set())
print(dataset[3])

batchsampler = ContrastBatchSampler(dataset, 10, 1)
dataloader = DataLoader(dataset, batch_sampler=batchsampler, num_workers=0, )
