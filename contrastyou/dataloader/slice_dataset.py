import argparse
import os
import re
import warnings
from itertools import repeat
from multiprocessing import Pool
from pprint import pprint
from typing import List, Tuple

import numpy as np
from nibabel import Nifti1Image
from nilearn.image import load_img
from nilearn.image import resample_img as _resample_img
from skimage.io import imsave
from sklearn.model_selection import train_test_split

from deepclustering2.utils import T_path, path2Path, path2str

target_affine = np.array([[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 10]])
common_afine = np.array([[-1.5625, 0., 0., 0],
                         [0., 1.5625, 0., 0],
                         [0., 0., 10., 0],
                         [0, 0, 0, 1]])
CROP_SIZE = [384, 384]
SAVE_DIR = "../../.data/ACDC_contrast"


class CenterCrop:
    """
    centercrop for ndarray
    """

    def __init__(self, crop_size=CROP_SIZE) -> None:
        self._crop_size = crop_size

    def __call__(self, input_array: np.ndarray):
        w, h, s = input_array.shape
        [xs, xe] = w // 2 - self._crop_size[0] // 2, w // 2 + self._crop_size[0] // 2
        [ys, ye] = h // 2 - self._crop_size[1] // 2, h // 2 + self._crop_size[1] // 2

        return input_array[xs:xe, ys:ye]


def array_normalize(array: np.ndarray, percentile: float) -> np.ndarray:
    array = array.astype(np.float)
    assert 0 < percentile < 1 and isinstance(percentile, float), percentile
    min_value = np.percentile(array, (1 - percentile) * 100)
    max_value = np.percentile(array, percentile * 100)
    array[array < min_value] = min_value
    array[array > max_value] = max_value
    nor_array = (array - array.min()) / (array.max() - array.min())
    assert np.alltrue(nor_array <= 1) and np.alltrue(nor_array >= 0)
    return nor_array


def get_patient_frames(folder_path: T_path) -> List[Tuple[Nifti1Image, Nifti1Image]]:
    frame_image_reg = r"patient\d+_frame\d+.nii.gz"
    frame_gt_reg = r"patient\d+_frame\d+_gt.nii.gz"
    affine_ref = r"patient\d+_4d.nii.gz"
    folder_path = path2str(folder_path)
    nii_list = sorted([x for x in os.listdir(folder_path) if re.compile(frame_image_reg).match(x)])
    nii_gt_list = sorted([x for x in os.listdir(folder_path) if re.compile(frame_gt_reg).match(x)])
    affine_list = sorted([x for x in os.listdir(folder_path) if re.compile(affine_ref).match(x)])
    assert len(nii_list) == len(nii_gt_list)
    fourd_matrix = load_img(os.path.join(folder_path, affine_list[0]))
    affine_matrix = fourd_matrix.affine
    try:
        assert affine_matrix[2][2] == 10.0
    except AssertionError:
        print(affine_matrix, fourd_matrix.shape)
        h, w, s, *_ = fourd_matrix.shape
        if h > 256 and s > 256:
            affine_matrix[2][2] = 10.0
        else:
            affine_matrix = common_afine

    return_nii = [(load_img(os.path.join(folder_path, x)),
                   load_img(os.path.join(folder_path, y))) for x, y in zip(nii_list, nii_gt_list)]
    for pairs in return_nii:
        for p in pairs:
            p.__dict__["_affine"] = affine_matrix
    return return_nii


def resample_nift(source_nift: Nifti1Image, target_affine: np.ndarray = None, target_nift: Nifti1Image = None,
                  interpolation="continuous"):
    assert isinstance(target_affine, np.ndarray) or isinstance(target_nift, Nifti1Image)
    assert not (isinstance(target_affine, np.ndarray) and isinstance(target_nift, Nifti1Image))
    if target_affine is None:
        target_affine = target_nift.affine
    source_resampled = _resample_img(source_nift, target_affine, interpolation=interpolation)
    return source_resampled


def write2png(array: np.ndarray, save_folder: T_path, patient_num: str, is_label=False, ):
    assert len(array.shape) == 3, array.shape
    if not is_label:
        assert ((0 <= array) & (array <= 1)).all(), (array.min(), array.max())
    png_array = (array * 255).astype(np.uint8)

    if is_label:
        png_array = array.astype(np.uint8)

    save_folder.mkdir(exist_ok=True, parents=True)

    for i in range(png_array.shape[2]):
        cur_img = png_array[:, :, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            imsave(f"{str(save_folder)}/{patient_num}_{i:02d}.png", arr=cur_img)


def padding_crop(img_array: np.ndarray, gt_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert img_array.shape == gt_array.shape, img_array.shape
    w, h, s = img_array.shape
    gt_per_s = gt_array.sum(axis=0).sum(axis=0)
    # remove slices without annotations
    img_array, gt_array = img_array[:, :, gt_per_s > 0], gt_array[:, :, gt_per_s > 0]
    # padding with 0 values
    img_array_pad = np.pad(img_array, pad_width=[[128, 128], [128, 128], [0, 0]], constant_values=0.0)
    gt_array_pad = np.pad(gt_array, pad_width=[[128, 128], [128, 128], [0, 0]], constant_values=0.0)
    centercrop = CenterCrop()
    return centercrop(img_array_pad), centercrop(gt_array_pad)


def patient_pipeline(patient_folder: T_path, save_folder: T_path):
    data_pair1, data_pair2 = get_patient_frames(patient_folder)

    def resample_pair(data_pair):
        img_resample = resample_nift(data_pair[0], target_affine=target_affine, )
        gt_resample = resample_nift(data_pair[1], target_affine=target_affine, interpolation="nearest")
        # img_resample = data_pair[0]
        # gt_resample = data_pair[1]
        return img_resample, gt_resample

    (img1_resample, gt1_resample), (img2_resample, gt2_resample) = resample_pair(data_pair1), resample_pair(data_pair2)
    img1_resample_nor, img2_resample_nor = array_normalize(img1_resample.get_fdata(dtype=np.float), percentile=0.99), \
                                           array_normalize(img2_resample.get_fdata(dtype=np.float), percentile=0.99)
    img1_resample_nor_crop, gt1_resample_crop = padding_crop(img1_resample_nor, gt1_resample.get_fdata())
    img2_resample_nor_crop, gt2_resample_crop = padding_crop(img2_resample_nor, gt2_resample.get_fdata())
    assert img1_resample_nor_crop.shape[:2] == tuple(CROP_SIZE)
    assert img2_resample_nor_crop.shape[:2] == tuple(CROP_SIZE)

    save_folder = path2Path(save_folder)
    write2png(img1_resample_nor_crop, save_folder=save_folder / "img",
              patient_num=path2Path(patient_folder).stem + "_00",
              is_label=False)
    write2png(img2_resample_nor_crop, save_folder=save_folder / "img",
              patient_num=path2Path(patient_folder).stem + "_01",
              is_label=False)
    write2png(gt1_resample_crop, save_folder=save_folder / "gt", patient_num=path2Path(patient_folder).stem + "_00",
              is_label=True)
    write2png(gt2_resample_crop, save_folder=save_folder / "gt", patient_num=path2Path(patient_folder).stem + "_01",
              is_label=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="nift data folder")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    root_dir = path2Path(args.root_dir)
    assert root_dir.exists() and root_dir.is_dir(), root_dir
    patient_list = sorted([x for x in root_dir.glob("*") if x.is_dir()])
    print(f"Found {len(patient_list)} patients:")
    pprint(patient_list[:5])
    train_patients, val_patients = train_test_split(patient_list, test_size=0.125)
    with Pool(8) as pool:
        pool.starmap(patient_pipeline, zip(train_patients, repeat(SAVE_DIR + "/train")))
    with Pool(8) as pool:
        pool.starmap(patient_pipeline, zip(val_patients, repeat(SAVE_DIR + "/val")))
    # patient_pipeline(patient_list[56], save_folder=SAVE_DIR)


#
def read_affine(folder_path):
    affine_ref = r"patient\d+_4d.nii.gz"
    folder_path = path2str(folder_path)
    affine_list = sorted([x for x in os.listdir(folder_path) if re.compile(affine_ref).match(x)])
    fourd_matrix = load_img(os.path.join(folder_path, affine_list[0]))
    return fourd_matrix.affine


if __name__ == '__main__':
    # root_dir = "./"
    # root_dir = path2Path(root_dir)
    # assert root_dir.exists() and root_dir.is_dir(), root_dir
    # patient_list = sorted([x for x in root_dir.glob("*") if x.is_dir()])
    # print(f"Found {len(patient_list)} patients:")
    # pprint(patient_list[:5])
    # for p in patient_list:
    #     print(p, read_affine(p))
    main()
