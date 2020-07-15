from deepclustering2.type import to_device


def preprocess_input_with_twice_transformation(labeled_input, unlabeled_input, device):
    [(labelimage, labeltarget), (labelimage_tf, labeltarget_tf)], filename, partition_list, group_list = \
        to_device(labeled_input[0], device), labeled_input[1], labeled_input[2], labeled_input[3]
    unlabelimage, unlabelimage_tf = to_device([unlabeled_input[0][0][0], unlabeled_input[0][1][0]],
                                              device)
    return (labelimage, labeltarget), (labelimage_tf, labeltarget_tf), filename, partition_list, group_list, (
        unlabelimage, unlabelimage_tf)


def preprocess_input_with_once_transformation(data, device):
    return data[0][0].to(device), data[0][1].to(device), data[1], data[2], data[3]


def preprocess_input_train_fs(data, device):  # noqa
    return data[0][0][0].to(device), data[0][0][1].to(device), data[1], data[2], data[3]