from contrastyou.data.dataset.acdc import ACDCDataset as _ACDCDataset
from contrastyou.data.sampler import ScanBatchSampler
from contrastyou.meters.general_dice_meter import UniversalDice
from torch.utils.data import DataLoader
import numpy as np
import torch

class ACDCDataset(_ACDCDataset):
    folder_name = ""
    sub_folders = [""]
    sub_folder_types = ["gt"]
    

    
def mapping(x):
    def _mapping(x):
        dict_ = dict([(0, 0), (9, 1), (35, 2), (36, 3)])
        try:
            return dict_[x]
        except:
            return 0
    
    return np.vectorize(_mapping)(x)

gt_dataset = ACDCDataset(mode="gt", root_dir="review_iic/PREDICT_MAPPING_IIC", transforms=None,)
batch_sampler = ScanBatchSampler(dataset=gt_dataset, shuffle=False, is_infinite=False)
gt_loader = DataLoader(gt_dataset, batch_sampler=batch_sampler, batch_size=1)


predict_dataset = ACDCDataset(mode="predict", root_dir="review_iic/PREDICT_MAPPING_IIC", transforms=None,)
batch_sampler = ScanBatchSampler(dataset=predict_dataset, shuffle=False, is_infinite=False)
pred_loader = DataLoader(predict_dataset, batch_sampler=batch_sampler, batch_size=1)

meter = UniversalDice(C=4, report_axis=[1,2, 3])

for cur_pred, cur_gt in zip(pred_loader, gt_loader):
    pred_class = torch.from_numpy(mapping(cur_pred[""]))
    gt_class = cur_gt[""]
    
    meter.add((pred_class).squeeze(1), (gt_class).squeeze(1))
    
print(meter.summary())
    
    
    



