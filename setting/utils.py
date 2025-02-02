import torch

def create_folder(save_path):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Create Folder [“{save_path}”].")
    return save_path
    
def compute_accuracy(predictions, ground_truth):
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = torch.sum(predicted_classes == ground_truth).item() / len(ground_truth)
    return accuracy

def random_seed_setting(seed: int = 42):
    """fixed random seed"""
    import random
    import os
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)