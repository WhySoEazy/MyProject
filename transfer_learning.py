import torch
from torchvision.models import resnet50 , ResNet50_Weights
import torch.nn as nn

class MyResnet50(nn.Module):
    def __init__(self , num_classes = 2):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.model.fc
        self.fc1 = nn.Linear(in_features=2048 , out_features=1024)
        self.fc2 = nn.Linear(in_features=1024 , out_features=num_classes)

    def _forward_impl(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == "__main__":

    image = torch.rand(2 , 3 , 224 , 224)
    model = MyResnet50()
    output = model(image)
    print(output.shape)