acdc_ratios = [0.01, 0.015, 0.025, 1.0]
# acdc_ratios = [0.01, 1.0]

prostate_ratio = [0.05, 0.1, 0.2, 1.0]  # 2 4 8
# prostate_ratio = [0.05, 1.0]  # 2 4 8

mmwhs_ratio = [0.09, 0.17, 0.34, 1.0]
# mmwhs_ratio = [0.09, 1.0]

ratio_zoom = {
    "acdc": acdc_ratios,
    "prostate": prostate_ratio,
    "mmwhsct": mmwhs_ratio,
    "mmwhsmr": mmwhs_ratio
}
dataset_name2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "spleen": 2,
    "mmwhs": 5,
}
ft_lr_zooms = {"acdc": 0.0000001,
               "prostate": 0.0000005,
               "spleen": 0.000001,
               "mmwhs": 0.000001}
pre_lr_zooms = {"acdc": 0.0000005, "prostate": 0.0000005, "mmwhs": 0.0000005}
# CC things

__accounts = ["rrg-mpederso", ]
# __accounts = ["def-chdesa", "rrg-mpederso", "def-mpederso"]
