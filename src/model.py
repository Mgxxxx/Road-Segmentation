import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import FCN_ResNet50_Weights
fcn = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)

class SPINModule(nn.Module):
    def __init__(self, in_channels):
        super(SPINModule, self).__init__()

        # Spatial Space Graph Reasoning
        self.spatial_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.diag_softmax = nn.Softmax(dim=-1)

        # Interaction Space Graph Reasoning
        self.inter_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.reverse_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Attention mechanism
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial reasoning
        spatial_out = self.spatial_proj(x)
        spatial_flat = spatial_out.view(x.shape[0], x.shape[1], -1)
        spatial_norm = self.diag_softmax(torch.matmul(spatial_flat, spatial_flat.transpose(1, 2)))
        spatial_proj = torch.matmul(spatial_norm, spatial_flat).view_as(spatial_out)

        # Interaction space reasoning
        inter_out = self.inter_proj(x)
        inter_out = self.reverse_proj(inter_out)

        # Spatial Attention
        attn_map = self.spatial_attn(x)
        output = spatial_proj * attn_map + inter_out

        return output

class SPINPyramid(nn.Module):
    def __init__(self, in_channels):
        super(SPINPyramid, self).__init__()
        self.spin1 = SPINModule(in_channels)
        self.spin2 = SPINModule(in_channels)
        self.spin3 = SPINModule(in_channels)
        self.weights = nn.Parameter(torch.ones(3))  # Learnable fusion weights

    def forward(self, x):
        original_size = x.shape[2:]
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        out1 = self.spin1(x)
        out2 = F.interpolate(self.spin2(x_half), size=original_size, mode='bilinear')
        out3 = F.interpolate(self.spin3(x_quarter), size=original_size, mode='bilinear')

        return self.weights[0] * out1 + self.weights[1] * out2 + self.weights[2] * out3


class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: List of channels for the backbone outputs [256, 512, 1024, 2048]
            out_channels: Number of output channels for the lateral convolutions
        """
        super(FPNDecoder, self).__init__()
        # Lateral convolutions to reduce backbone channels to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        
        # Output convolutions to refine the upsampled features
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from the backbone
                      (e.g., [layer1, layer2, layer3, layer4])
        """
        # Start with the highest level feature
        last_inner = self.lateral_convs[-1](features[-1])  # Reduce channels
        outputs = [self.output_convs[-1](last_inner)]      # Refine feature

        # Process the remaining features in reverse order
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])  # Reduce channels
            last_inner = F.interpolate(last_inner, size=lateral.shape[2:], mode="nearest") + lateral
            outputs.insert(0, self.output_convs[i](last_inner))  # Refine and store output



        return outputs  # Multi-scale feature maps (FPN output)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Fix shortcut logic: Only apply 1x1 conv if channels are different
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)  # Shortcut connection
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BN + ReLU
        x = self.bn2(self.conv2(x))  # Conv2 + BN
        return F.relu(x + residual)  # Add shortcut + activation


class HourglassModule(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(HourglassModule, self).__init__()

        # Project input to mid_channels
        self.input_proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        # Downsampling Path
        self.down1 = ResidualBlock(mid_channels, mid_channels)
        self.down2 = ResidualBlock(mid_channels, mid_channels)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(mid_channels, mid_channels)

        # Upsampling Path
        self.up1 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Project input to mid_channels (64 channels)
        x_proj = self.input_proj(x)  # Ensures x_proj is 64 channels

        # Downsampling
        x1 = self.down1(x_proj)
        x2 = self.down2(self.pool(x1))

        # Bottleneck
        x3 = self.bottleneck(x2)

        # Upsampling + Skip Connections
        x4 = self.up1(x3) + x1  # Skip connection from down1
        
        # Upsample input projection to match x4 spatial dimensions
        x_proj_upsampled = F.interpolate(x_proj, size=self.up2(x4).shape[2:], mode='bilinear', align_corners=True)

        x5 = self.up2(x4) + x_proj_upsampled  # Final upsample + skip connection from input projection

        return x5


class SPINRoadMapperFCN8(nn.Module):
    def __init__(self):
        super(SPINRoadMapperFCN8, self).__init__()
        # Replace FCN backbone with Hourglass backbone
        self.feature_extractor = HourglassModule(in_channels=3, mid_channels=64)

        # SPIN Pyramid
        self.spin_pyramid = SPINPyramid(in_channels=64)

        # Segmentation head
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)
        self.orientation_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Step 1: Extract features using Hourglass backbone
        features = self.feature_extractor(x)

        # Step 2: SPIN Pyramid for multi-scale reasoning
        spin_out = self.spin_pyramid(features)

        # Step 3: Segmentation and orientation outputs
        seg_output = self.segmentation_head(spin_out)
        orientation_output = self.orientation_head(spin_out)

        # Step 4: Upsample to match input size
        seg_output = F.interpolate(seg_output, size=x.shape[2:], mode="bilinear", align_corners=True)
        orientation_output = F.interpolate(orientation_output, size=x.shape[2:], mode="bilinear", align_corners=True)

        return seg_output, orientation_output