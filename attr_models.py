import torch
import torch.nn as nn
import torchvision
from torchvision import models
    
def resnext101(attr_labels):
    resnext_features = torchvision.models.resnext101_64x4d(weights='ResNeXt101_64X4D_Weights.DEFAULT')
    resnext_features.requires_grad_(False)
    resnext_features.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, len(attr_labels)),
        nn.Sigmoid()
    )
    resnext_features.fc.requires_grad_(True)
    return resnext_features

    
class Resnext101_model:
    def __init__(self, path_to_weight, attr_labels, device):
        self.device = device
        self.attr = attr_labels
        self.model_resnext = resnext101(attr_labels).to(self.device)
        self.model_resnext.load_state_dict(torch.load(path_to_weight, map_location=torch.device(device)))
        self.model_resnext.eval()

    def run(self, batch):
        answer = []
        pred = self.model_resnext(batch)
        acc = pred.data.tolist()
        for acci in acc:
            answer.append(self.attr[acci.index(max(acci))])
        return answer
        
class Garbage_model(Resnext101_model):
    def __init__(self, device):
        super().__init__('weights/garbage_weights.pt', ['bulk', 'KGM', 'TKO'], device)
        
class Fullness_model(Resnext101_model):
    def __init__(self, device):
        super().__init__('weights/fullness_weights.pt', ['empty', 'half', 'full'], device)
        
class Damage_model(Resnext101_model):
    def __init__(self, device):
        super().__init__('weights/damage_weights.pt', ['broken', 'flipped', 'ok'], device)