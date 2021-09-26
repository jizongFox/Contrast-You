import torch


def load_prototype_vector(model_path, *, device="cpu", dtype: torch.float):
    state_dict = torch.load(model_path, map_location=device)
    model_dict = state_dict["_model"]
    prototype_kernel =None
    return prototype_kernel



if __name__ == '__main__':
    load_prototype_vector()
