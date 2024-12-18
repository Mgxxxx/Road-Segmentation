import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
    
# class DownSample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = DoubleConv(in_channels, out_channels)
#         self.pool = nn.MaxPool2d(2, 2)
        
#     def forward(self, x):
#         down = self.conv(x)
#         p = self.pool(down)
        
#         return down, p
    

# class UpSample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_channels, out_channels)
        
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
        
#         return self.conv(x)
        
        