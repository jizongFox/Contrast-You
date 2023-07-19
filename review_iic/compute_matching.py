from PIL import Image, ImageFile
import typing as t
from pathlib import Path

import torch


from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
#   assert (isinstance(flat_preds, torch.Tensor) and
#           isinstance(flat_targets, torch.Tensor) and
#           flat_preds.is_cuda and flat_targets.is_cuda)

  num_samples = flat_targets.shape[0]

  num_correct = np.zeros((preds_k,targets_k))

  for c1 in range(preds_k):
    for c2 in range(targets_k):
      # elementwise, so each sample contributes once
      votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
      num_correct[c1, c2] = votes

  # num_correct is small
  match = linear_assignment(- num_correct)

  # return as list of tuples, out_c to gt_c

  return list(zip(*match))

def _original_match(flat_preds, flat_targets, preds_k, targets_k):
    # map each output channel to the best matching ground truth (many to one)

    # assert (isinstance(flat_preds, torch.Tensor) and
    #         isinstance(flat_targets, torch.Tensor) and
    #         flat_preds.is_cuda and flat_targets.is_cuda)
    out_to_gts = {}
    out_to_gts_scores = {}
    for out_c in range(preds_k):
        for gt_c in range(targets_k):
            # the amount of out_c at all the gt_c samples
            tp_score = int(
                ((flat_preds == out_c) * (flat_targets == gt_c)).sum())
            if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_scores[out_c] = tp_score

    return list(out_to_gts.items())

def read_tra_image(file_paths:t.List[Path])->t.Tuple[np.ndarray, np.ndarray]:
    def read_pil_to_numpy_flat(filepath:Path)->np.ndarray:
        assert filepath.exists()
        with Image.open(filepath) as f:
            img = f.convert("L")
        img_np = np.array(img).flatten()
        return img_np
    
    result = [read_pil_to_numpy_flat(x) for x in file_paths]
    return np.concatenate(result)

def _mapping(x):
    dict_ = dict([(0, 0), (9, 1), (35, 2), (36, 3)])
    try:
        return dict_[x]
    except:
        return 0
    
def mapping(x):
    return np.vectorize(_mapping)(x)

if __name__ =="__main__":
    train_list = sorted(Path("review_iic/PREDICT_MAPPING_IIC/predict").glob("*png"))
    test_list = sorted(Path("review_iic/PREDICT_MAPPING_IIC/gt").glob("*png"))
    import matplotlib.pyplot as plt
    
    def read_np(filepath):
        with Image.open(filepath) as f:
            img = f.convert("L")
        img_np = np.array(img)
        return img_np
    
    tra_image = read_np(train_list[0])
    tra_gt = read_np(test_list[0])
    
    plt.figure()
    plt.imshow(tra_image)
    plt.savefig("1.png")
    
    plt.figure()
    plt.imshow(mapping(tra_image))
    plt.savefig("2.png")
    plt.figure()
    plt.imshow(tra_gt)
    plt.savefig("3.png")
    # plt.show()
    
    
    
    # assert [x.name for x in train_list] == [x.name for x in test_list]
    # pred_list = read_tra_image(sorted(Path("review_iic/PREDICT_MAPPING_IIC/predict").glob("*png")))
    # target_list = read_tra_image(sorted(Path("review_iic/PREDICT_MAPPING_IIC/gt").glob("*png")))
    
    # match = _hungarian_match(pred_list, target_list, 40, 4)
    # print(match)
    # match = _original_match(pred_list, target_list, 40, 4)
    # print(match)
    
    # breakpoint()
