import torch
import torch.nn as nn
from torchsummary import summary

'''
DW: Depthwise separable convolution
    Theory" 
        EX) 3 X 3 X 3 입력에 대하여 Conv 연산후  3 X 3 X 3 출력을 하는 경우.
        (1) 보통의 방법
            Origin Image = 3 X 3 X 3 (H, W, C)
            아웃풋의 채널이 3이므로 커널의 크기는 3이고 이에 따라 
            Origin Image x Size of Kernel = 3 X 3 X 3 X 3 = 81 
            Result to Calculate Conv param = 81
        (2) DWConv의 방법
            Origin Image = 3 X 3 X 3 (H, W, C)
            아웃풋의 채널이 3이므로 커널의 크기는 3이고 이에 따라 
            Depthwise Separable Conv: 3 x 3 x 1 x 3 + 1 X 1 X 3 X 3 = 27 + 9 = 36
            Depthwise Conv = 3 x 3 x 1 x 3 (H, W, C, num of Kernel)
            Pointwise Conv = 1 x 1 x 3 x 3 (H, W, C, num of Kernel)
            Result to Calculate Conv param = 36
'''
class MobileNet(nn.Module):
    def __init__(self): 
        super(MobileNet, self).__init__()
        def InitConv():
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),            
            )    

        def DWConv(in_ch, out_ch, dw_stride):
            return nn.Sequential(
                #dw
                nn.Conv2d(in_ch, in_ch, 3, dw_stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),

                #pw
                nn.Conv2d(in_ch, out_ch, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),            
            )
    
        self.model = nn.Sequential(
            InitConv(),
            DWConv(32, 64, 1),
            DWConv(64, 128, 2),
            DWConv(128, 128, 1),
            DWConv(128, 256, 2),
            DWConv(256, 256, 1),
            DWConv(256, 512, 2),
            DWConv(512, 512, 1),
            DWConv(512, 512, 1),
            DWConv(512, 512, 1),
            DWConv(512, 512, 1),
            DWConv(512, 512, 1),
            DWConv(512, 1024, 2),
            DWConv(1024, 1024, 1),
            nn.AvgPool2d(1)        
        )
        self.fc = nn.Linear(1024, 1000)
    
    def forward(self,x):
        out = self.model(x)  
        out = out.view(-1, 1024)
        out = self.fc(out)      
        return out
    
device = torch.device('cpu')

if __name__ == "__main__":
    model = MobileNet()
    summary(model=model, input_size= (3,224,224), device= device.type)
