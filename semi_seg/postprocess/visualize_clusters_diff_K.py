import re
import typing as t
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from torchvision.transforms import CenterCrop

from contrastyou.utils.colors import label2colored_image


def image_grouper(root_dir: str, pattern: str) -> t.Iterator:
    def load_np_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def process(ndimage: np.ndarray):
        return CenterCrop(224)(torch.from_numpy(ndimage)).numpy()

    png_sorted_list = sorted(Path(root_dir).rglob("*.png"))
    grex = re.compile(pattern)
    group_names = []
    for image in png_sorted_list:
        matched = grex.match(str(image.relative_to(root_dir)))
        if matched:
            group_names.append(matched.group())
    assert len(group_names) > 0, group_names
    group_names = sorted(set(group_names))
    for g in group_names:
        filenames = sorted([x for x in png_sorted_list if grex.match(str(x.relative_to(root_dir))).group() == g])
        yield np.stack([process(load_np_image(x)) for x in filenames])


def cc_grouper(root_dir: str, pattern: str) -> t.Iterator:
    npy_list = sorted(Path(root_dir).rglob("*.npy"))
    grex = re.compile(pattern)
    for npy in npy_list:
        matched = grex.match(str(npy.relative_to(root_dir)))
        if matched:
            yield np.load(npy).argmax(1)


def superpixel_grouper(root_dir: str, group_pattern: str) -> t.Iterator:
    npy_sorted_list = sorted(Path(root_dir).rglob("*.npy"))
    grex = re.compile(group_pattern)
    group_names = []
    for image in npy_sorted_list:
        matched = grex.match(str(image.relative_to(root_dir)))
        if matched:
            group_names.append(matched.group())
    assert len(group_names) > 0, group_names
    group_names = sorted(set(group_names))
    for g in group_names:
        filenames = sorted([x for x in npy_sorted_list if grex.match(str(x.relative_to(root_dir))).group() == g])
        yield np.stack([np.load(x) for x in filenames])


def cluster_alignment(cluster: np.ndarray, reference_cluster: np.ndarray):
    assert cluster.shape == reference_cluster.shape, (cluster.shape, reference_cluster.shape)


def _hungarian_match(
        flat_preds, flat_targets, preds_k, targets_k
):
    assert flat_preds.shape == flat_targets.shape

    num_samples = flat_targets.shape[0]

    assert preds_k == targets_k  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        match = linear_assignment(num_samples - num_correct)

    # return as list of tuples, out_c to gt_c
    res = {}
    for out_c, gt_c in zip(*match):
        # res.append((out_c, gt_c))
        res[out_c] = gt_c
    return res


def hungarian_match(*cluster: np.ndarray, reference_cluster: np.ndarray, num_clusters: int):
    new_cluster_list = []
    for cur_cluster in cluster:
        mapping = _hungarian_match(cur_cluster.flatten(), reference_cluster.flatten(), num_clusters, num_clusters)
        new_cluster = deepcopy(cur_cluster)
        for c in mapping:
            new_cluster[cur_cluster == c] = mapping[c]
        new_cluster_list.append(new_cluster)
    return tuple(new_cluster_list)


def get_segment(iter_: t.Iterator, num_volume: int):
    segment = iter_.__next__()
    for _ in range(num_volume):
        segment = iter_.__next__()
    return segment


if __name__ == '__main__':
    image_dir = "/home/jizong/Workspace/Contrast-You/.data/ACDC_contrast/train/img"
    gt_dir = "/home/jizong/Workspace/Contrast-You/.data/ACDC_contrast/train/gt"
    k_05 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_dc984e0f99f/acdc/pretrain/seed_10/cc_weight_1/pretrain_scan_sample_num_6/consistency_weight_0/rr_weight_1/power_0.75/head_type_linear/num_subheads_3/kernel_size_5/rr_symmetric_true/num_clusters_5/rr_lamda_1/rr_alpha_0.5/pretrain/matrix/cc_Up_conv2"
    k_10 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_dc984e0f99f/acdc/pretrain/seed_10/cc_weight_1/pretrain_scan_sample_num_6/consistency_weight_0/rr_weight_1/power_0.75/head_type_linear/num_subheads_3/kernel_size_5/rr_symmetric_true/num_clusters_10/rr_lamda_1/rr_alpha_0.5/pretrain/matrix/cc_Up_conv2"
    k_20 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_dc984e0f99f/acdc/pretrain/seed_10/cc_weight_1/pretrain_scan_sample_num_6/consistency_weight_0/rr_weight_1/power_0.75/head_type_linear/num_subheads_3/kernel_size_5/rr_symmetric_true/num_clusters_20/rr_lamda_1/rr_alpha_0.5/pretrain/matrix/cc_Up_conv2"
    k_40 = "/home/jizong/Workspace/Contrast-You/runs/beluga/1224_combine/acdc/hash_955663565fd/acdc/pretrain/sample/rr_alpha_0.5/pretrain/matrix/cc_Up_conv2"
    k_60 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_dc984e0f99f/acdc/pretrain/seed_10/cc_weight_1/pretrain_scan_sample_num_6/consistency_weight_0/rr_weight_1/power_0.75/head_type_linear/num_subheads_3/kernel_size_5/rr_symmetric_true/num_clusters_60/rr_lamda_1/rr_alpha_0.5/pretrain/matrix/cc_Up_conv2"
    # superpixel_dir = "/home/jizong/Workspace/Contrast-You/.data/ACDC_superpixel/train/img"

    # /home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_2d42fc6b083/prostate/sample

    # image_dir = "/home/jizong/Workspace/Contrast-You/.data/PROSTATE/train/img"
    # gt_dir = "/home/jizong/Workspace/Contrast-You/.data/PROSTATE/train/gt"
    # rr_050 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_2d42fc6b083/prostate/sample/rr_alpha_0.5/pretrain/matrix/cc_Up_conv2"
    # rr_075 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_2d42fc6b083/prostate/sample/rr_alpha_0.75/pretrain/matrix/cc_Up_conv2"
    # rr_025 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_2d42fc6b083/prostate/sample/rr_alpha_0.25/pretrain/matrix/cc_Up_conv2"
    # rr_000 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_2d42fc6b083/prostate/sample/rr_alpha_0/pretrain/matrix/cc_Up_conv2"
    # rr_100 = "/home/jizong/Workspace/Contrast-You/runs/narval/0107/cc/hash_2d42fc6b083/prostate/sample/rr_alpha_1/pretrain/matrix/cc_Up_conv2"
    # superpixel_dir = "/home/jizong/Workspace/Contrast-You/.data/PROSTATE_superpixel/train/img"

    pattern = "probability"
    slice_index = 2
    volume_num = 5

    image_gen = image_grouper(root_dir=image_dir, pattern=r"patient\d+_\d+")
    segment = get_segment(image_gen, volume_num)
    image_slice = segment[slice_index]
    # 
    gt_gen = image_grouper(root_dir=gt_dir, pattern=r"patient\d+_\d+")
    segment = get_segment(gt_gen, volume_num)
    gt_slice = segment[slice_index]

    # superpixel = superpixel_grouper(superpixel_dir, r"Case\d+_\d+")
    # segment = get_segment(superpixel, volume_num)
    # slice_sp = segment[slice_index]

    case_05 = cc_grouper(k_05, pattern=f"{pattern}_\d+\d+")
    segment = get_segment(case_05, volume_num)
    slice_05 = segment[slice_index]

    case_10 = cc_grouper(k_10, pattern=f"{pattern}_\d+\d+")
    segment = get_segment(case_10, volume_num)
    slice_10 = segment[slice_index]

    case_20 = cc_grouper(k_20, pattern=f"{pattern}_\d+\d+")
    segment = get_segment(case_20, volume_num)
    slice_20 = segment[slice_index]

    case_40 = cc_grouper(k_40, pattern=f"{pattern}_\d+\d+")
    segment = get_segment(case_40, volume_num)
    slice_40 = segment[slice_index]

    case_60 = cc_grouper(k_60, pattern=f"{pattern}_\d+\d+")
    segment = get_segment(case_60, volume_num)
    slice_60 = segment[slice_index]

    slice_05, slice_10, slice_20, slice_40, slice_60, = \
        hungarian_match(slice_05, slice_10, slice_20, slice_40, slice_60,
                        reference_cluster=slice_40,
                        num_clusters=60)
    plt.subplot(331)
    plt.imshow(image_slice, cmap="gray")
    plt.axis('off')

    plt.subplot(332)
    plt.imshow(label2colored_image(slice_05))
    plt.axis('off')

    plt.subplot(333)
    plt.imshow(label2colored_image(slice_10))
    plt.axis('off')

    plt.subplot(334)
    plt.imshow(label2colored_image(slice_20))
    plt.axis('off')

    plt.subplot(335)
    plt.imshow(label2colored_image(slice_40))
    plt.axis('off')

    plt.subplot(336)
    plt.imshow(label2colored_image(slice_60))
    plt.axis('off')

    plt.subplot(337)
    plt.imshow(gt_slice)
    plt.axis('off')
    plt.tight_layout()

    plt.show()
