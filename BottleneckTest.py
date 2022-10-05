# Example_of_the_difference_between_Standard_and_BottleNeck_parameters_in_ResNet

import torch
import torch.nn as nn

class Standard(nn.Module):
    def __init__(self, in_dim = 256, mid_dim = 64, out_dim = 64):
        super(Standard, self).__init__()
        self.building_block = nn.Sequential(
            nn.Conv2d(in_channels= in_dim, out_channels=mid_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=3, padding=False)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        fx = self.building_block(x)
        out = fx + x
        out = self.relu(out)
        return out
 
class BottleNeck(nn.Module):
    def __init__(self, in_dim = 256, mid_dim = 12, out_dim = 256):
        super(BottleNeck, self).__init__()
        self.building_block = nn.Sequential(
            nn.Conv2d(in_channels= in_dim, out_channels=mid_dim, kernel_size=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=3, padding=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_dim, out_channels=out_dim, kernel_size=1, padding=False)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        fx = self.building_block(x)
        out = fx + x
        out = self.relu(out)
        return out
       
def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
        
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
 

if __name__ == "__main__":
    print("\nStandard Param")
    test(Standard())
    print("\nBottleNeck Param")    
    test(BottleNeck())
