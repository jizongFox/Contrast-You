from pathlib import Path

import numpy as np
from skimage.io import imsave
from skimage.segmentation import slic
from torchvision import transforms
from tqdm import tqdm

from contrastyou import DATA_PATH
from contrastyou.augment import SequentialWrapper, pil_augment
from semi_seg.data import ACDCDataset

mode = "train"
output_dir = f"{mode}/superpixel"
transform = SequentialWrapper(
    image_transform=transforms.Compose([
        transforms.ToTensor()
    ]),
    target_transform=transforms.Compose([
        pil_augment.ToLabel()
    ]),
)
data = ACDCDataset(root_dir=DATA_PATH, mode=mode, transforms=transform, )

for i in tqdm(range(len(data))):
    (image, gt,), filenanme, _ = data[i]
    segments = slic(image[0], n_segments=40, sigma=5, compactness=0.1)
    save_path = Path(output_dir, filenanme + ".png")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(str(save_path), segments.astype(np.uint8))
