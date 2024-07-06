import torch.nn as nn
import torchvision

def ViT():
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=3)
    return pretrained_vit

