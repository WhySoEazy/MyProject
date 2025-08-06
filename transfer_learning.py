import torch
from torchvision.models import resnet50 , ResNet50_Weights
import torch.nn as nn
from torchsummary import summary

model = resnet50(ResNet50_Weights.DEFAULT)

model.fc = nn.Linear(in_features=2048 , out_features=2)
for name , params in model.named_parameters():
    if "fc." in name or "layer4." in name:
        pass
    else:
        params.requires_grad = False
    print(name , params.requires_grad)

summary(model , (3 , 224 , 224))