import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
    
class Standard_Param(nn.Module):
    def __init__(self):
        super(Standard_Param, self).__init__()
        
        self.Standard_Param = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        fx = self.Standard_Param(x)
        out = fx + x
        out = self.relu(out)
        return out
 
class BottleNeck_Param(nn.Module):
    def __init__(self, inputdim = 1, outputdim = 64):
        super(BottleNeck_Param, self).__init__()
        
        self.BottleNeck_Param = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=False),            
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        fx = self.BottleNeck_Param(x)
        out = fx + x
        out = self.relu(out)
        return out
               
    
device = torch.device('cpu')

batch_size = 1

if __name__ == "__main__":
    model1 = Standard_Param()
    model2 = BottleNeck_Param()
    summary(model=model1, input_size= (256,320,320), device= device.type)
    summary(model=model2, input_size= (256,320,320), device= device.type)


