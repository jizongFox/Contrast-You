from deepclustering2.type import to_device


def preprocess_input_with_twice_transformation(data, device):
    [(image, target), (image_tf, target_tf)], filename, partition_list, group_list = \
        to_device(data[0], device), data[1], data[2], data[3]
    return (image, target), (image_tf, target_tf), filename, partition_list, group_list


def preprocess_input_with_single_transformation(data, device):
    return data[0][0].to(device), data[0][1].to(device), data[1], data[2], data[3]
