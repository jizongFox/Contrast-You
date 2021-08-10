acdc_ratios = [1, 2, 4, 174]
prostate_ratio = [3, 5, 7, 40]  # 2, 4, 8, 40
mmwhsct_ratio = [1, 2, 10]
prostate_md_ratio = [1, 2, 4, 17]  # 1, 2, 4, 8
mmwhsmr_ratio = [1, 2, 9]
spleen_ratios = [2, 4]
hippocampus_ratios = [1, 2, 4]

pre_max_epoch_zoo = {
    "acdc": 80,
    "mmwhsct": 80,
    "mmwhsmr": 80,
    "prostate": 80,
    "spleen": 40,
    "hippocampus": 80
}
ft_max_epoch_zoo = {
    "acdc": 80,
    "mmwhsct": 60,
    "mmwhsmr": 60,
    "prostate": 80,
    "spleen": 80,
    "hippocampus": 80
}
num_batches_zoo = {
    "acdc": 200,
    "mmwhsct": 350,
    "mmwhsmr": 350,
    "prostate": 300,
    "spleen": 200,
    "hippocampus": 200
}

ratio_zoo = {
    "acdc": acdc_ratios,
    "prostate": prostate_ratio,
    "prostate_md": prostate_md_ratio,
    "mmwhsct": mmwhsct_ratio,
    "mmwhsmr": mmwhsmr_ratio,
    "spleen": spleen_ratios,
    "hippocampus": hippocampus_ratios
}
data2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "prostate_md": 3,
    "spleen": 2,
    "mmwhsct": 5,
    "mmwhsmr": 5,
    "hippocampus": 3

}
data2input_dim = {
    "acdc": 1,
    "prostate": 1,
    "prostate_md": 1,
    "spleen": 1,
    "mmwhsct": 1,
    "mmwhsmr": 1,
    "hippocampus": 1
}

pre_lr_zooms = {
    "acdc": 0.0000005,
    "prostate": 0.0000005,
    "prostate_md": 0.000005,
    "mmwhsct": 0.0000005,
    "mmwhsmr": 0.0000005,
    "spleen": 0.0000005,
    "hippocampus": 0.0000005,
}

ft_lr_zooms = {
    "acdc": 0.0000002,
    "prostate": 0.0000005,
    "prostate_md": 0.0000005,
    "spleen": 0.00000005,
    "mmwhsct": 0.000002,
    "mmwhsmr": 0.000002,
    "hippocampus": 0.0000002,
}
__accounts = ["rrg-mpederso", ]

PRETRAIN_BATCH_SIZE_MAX = 50
# __accounts = ["def-chdesa", "rrg-mpederso", "def-mpederso"]
