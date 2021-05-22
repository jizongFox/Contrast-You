from contrastyou import on_cc

acdc_ratios = [0.01, 1.0]
prostate_ratio = [0.075, 1.0]  # 2 4 8
prostate_md_ratio = [0.06, 1.0]
mmwhsct_ratio = [0.1, 1.0]
mmwhsmr_ratio = [0.12, 1.0]

if on_cc():
    acdc_ratios = [0.01, 0.015, 0.025, 1.0]
    prostate_ratio = [0.075, 0.125, 0.175, 1.0]  # 2, 4, 8, 40
    mmwhsct_ratio = [0.1, 0.2, 1.0]
    prostate_md_ratio = [0.06, 0.12, 0.24, 1.0]  # 1, 2, 4, 8
    mmwhsmr_ratio = [0.12, 0.24, 0.45, 1.0]

ratio_zoo = {
    "acdc": acdc_ratios,
    "prostate": prostate_ratio,
    "prostate_md": prostate_md_ratio,
    "mmwhsct": mmwhsct_ratio,
    "mmwhsmr": mmwhsmr_ratio,
}
data2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "prostate_md": 3,
    "spleen": 2,
    "mmwhsct": 5,
    "mmwhsmr": 5,

}
data2input_dim = {
    "acdc": 1,
    "prostate": 1,
    "prostate_md": 1,
    "spleen": 1,
    "mmwhsct": 1,
    "mmwhsmr": 1,
}

pre_lr_zooms = {
    "acdc": 0.0000005,
    "prostate": 0.0000005,
    "prostate_md": 0.000005,
    "mmwhsct": 0.0000005,
    "mmwhsmr": 0.0000005
}

ft_lr_zooms = {
    "acdc": 0.0000001,
    "prostate": 0.0000005,
    "prostate_md": 0.0000005,
    "spleen": 0.000001,
    "mmwhsct": 0.000002,
    "mmwhsmr": 0.000002
}
__accounts = ["rrg-mpederso", ]
# __accounts = ["def-chdesa", "rrg-mpederso", "def-mpederso"]

labeled_filenames = {
    "acdc": {1: ["patient100_00"],
             2: ["patient027_01", "patient100_00"],
             4: ["patient027_01", "patient038_01", "patient067_01", "patient100_00"]},
    "prostate": {3: ["Case10", "Case17", "Case45"],
                 5: ["Case00", "Case10", "Case17", "Case37", "Case45"],
                 7: ["Case00", "Case10", "Case17", "Case34", "Case37", "Case38", "Case45"]},
    "mmwhsct": {1: ["1003"],
                2: ["1003", "1010"]}
}