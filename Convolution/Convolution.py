import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
    
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=55, kernel_size=11, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=55, out_channels=256, kernel_size=5, padding=2, bias=True),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.FC = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features= 9216, out_features= 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features= 4096, out_features= 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features= 4096, out_features= 1000),            
        )
        
        self.Init_Bias()
    
    def Init_Bias(self):
        for layer in self.Layer:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std= 0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.Layer[4].bias, 1)
        nn.init.constant_(self.Layer[10].bias, 1)
        nn.init.constant_(self.Layer[12].bias, 1)
    
    def forward(self, x):
        x = self.Layer(x)
        x = x.view(-1,256 * 6 *6)
        out = self.FC(x)
        return out


device = torch.device('cpu')

batch_size = 1

if __name__ == "__main__":
    model1 = AlexNet()
    summary(model=model1, input_size= (3,227,227), device= device.type)


