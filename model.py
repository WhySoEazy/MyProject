import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnet50 , ResNet50_Weights

class SimpleCNN(nn.Module):
    def __init__(self , num_classes = 10):

        super().__init__()
        
        self.conv1 = self._make_block(in_channels=3 , out_channels=8)
        self.conv2 = self._make_block(in_channels=8 , out_channels=16)
        self.conv3 = self._make_block(in_channels=16 , out_channels=32)
        self.conv4 = self._make_block(in_channels=32 , out_channels=64)
        self.conv5 = self._make_block(in_channels=64 , out_channels=128)
        self.flatten = nn.Flatten()
        
        self.FullyConnectedLayer1 = nn.Sequential(
            nn.Linear(in_features=6272 , out_features=512),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.FullyConnectedLayer2 = nn.Sequential(
            nn.Linear(in_features=512 , out_features=1024),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.FullyConnectedLayer3 = nn.Sequential(
            nn.Linear(in_features=1024 , out_features=num_classes),
            nn.Dropout(p=0.5)
        )

        model = resnet50(ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=2048 , out_features=2)
        for name , params in self.named_parameters():
            if "conv5." in name or "FullyConnectedLayer1." in name or "FullyConnectedLayer2." in name or "FullyConnectedLayer3." in name:
                pass
            else:
                params.requires_grad = False

    def _make_block(self , in_channels , out_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=3 , stride=1 , padding="same"),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels , out_channels=out_channels , kernel_size=3 , stride=1 , padding="same"),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )


    def forward(self , x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.FullyConnectedLayer1(x)
        x = self.FullyConnectedLayer2(x)
        x = self.FullyConnectedLayer3(x)
        # x = x.view(x.shape[0] , -1)
        # x = x.view(x.shape[0] , x.shape[1]*x.shape[2]*x.shape[3])
        return x

if __name__ == "__main__":
    model = SimpleCNN()
    # device = "gpu" if torch.cuda.is_available() else "cpu"
    # print(device)
    # input_data = torch.rand(8 , 3 , 224 , 224)
    # if torch.cuda.is_available():
    #     model.cuda()
    #     input_data = input_data.cuda()
    # result = model.forward(input_data)
    # print(result.shape)
    print(summary(model , (3 , 224 , 224)))

    # for name , params in model.named_parameters():
    #     print(name , params.shape)