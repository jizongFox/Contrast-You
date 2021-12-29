from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from skimage import io
from skimage.io import imsave
from skimage.segmentation import slic
from skimage.util import img_as_float
from torchvision.transforms import CenterCrop

from contrastyou import DATA_PATH


def do_superpixel(image_path: str, num_clusters: int):
    image = img_as_float(io.imread(str(image_path)))
    image = CenterCrop(224)(torch.from_numpy(image))
    # loop over the number of segments

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments=num_clusters, compactness=0.03, multichannel=False, sigma=5)
    return segments


def pipeline(image_path: str, num_clusters: int, root_dir: str, save_dir: str):
    relative_path = Path(image_path).relative_to(root_dir)
    save_path = Path(save_dir) / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    segment = do_superpixel(image_path=image_path, num_clusters=num_clusters)
    np.save(str(save_path), segment.astype(np.int16, copy=False))
    imsave(str(save_path), segment.astype(np.uint8))


if __name__ == '__main__':
    root_dir = Path(DATA_PATH, "ACDC_contrast")
    save_dir = Path(DATA_PATH, "ACDC_superpixel")
    image_list = sorted((root_dir / "train" / "img").glob("*.png"))

    pipeline_ = partial(pipeline, num_clusters=40, root_dir=root_dir, save_dir=save_dir)
    with Pool(16) as pool:
        pool.map(pipeline_, image_list)
