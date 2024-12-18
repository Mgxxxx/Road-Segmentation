import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import FCN_ResNet50_Weights
fcn = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)

# SPIN Module
class SPINModule(nn.Module):
    def __init__(self, in_channels):
        super(SPINModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.graph_layer = nn.Linear(in_channels, in_channels)  # Interaction space reasoning        
        # Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial reasoning
        spatial_out = self.conv1(x)

        # Interaction space reasoning
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1).transpose(1, 2)
        interaction_out = self.graph_layer(x_flat)
        interaction_out = interaction_out.transpose(1, 2).view(batch_size, channels, height, width)

        # Spatial Attention
        attn_map = self.spatial_attn(x)
        spatial_out = spatial_out * attn_map

        return spatial_out + interaction_out

class SPINPyramid(nn.Module):
    def __init__(self, in_channels):
        super(SPINPyramid, self).__init__()
        self.spin1 = SPINModule(in_channels)
        self.spin2 = SPINModule(in_channels)
        self.spin3 = SPINModule(in_channels)

    def forward(self, x):
        # Original input size
        original_size = x.shape[2:]

        # Downsample to half and quarter scales
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)

        # Apply SPIN Modules
        out1 = self.spin1(x)  # Original scale
        out2 = self.spin2(x_half)
        out3 = self.spin3(x_quarter)

        # Upsample back to the original size
        out2 = F.interpolate(out2, size=original_size, mode='bilinear', align_corners=True)
        out3 = F.interpolate(out3, size=original_size, mode='bilinear', align_corners=True)

        # Combine outputs
        return out1 + out2 + out3


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


class SPINRoadMapperFCN8(nn.Module):
    def __init__(self):
        super(SPINRoadMapperFCN8, self).__init__()
        # Load pretrained FCN-8 backbone
        fcn = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        self.backbone = fcn.backbone  # Use FCN's ResNet50 encoder
        
        # FPN Decoder - Expecting only the 'out' and 'aux' features
        self.fpn_decoder = FPNDecoder(in_channels_list=[2048, 1024], out_channels=256)
        
        # SPIN Pyramid
        self.spin_pyramid = SPINPyramid(in_channels=256)

        # Segmentation head
        self.segmentation_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # FPN Decoder takes 'out' and 'aux' feature maps
        fpn_features = self.fpn_decoder([features['out'], features['aux']])
        
        # Sum up FPN outputs to combine multi-scale features
        combined_features = torch.sum(torch.stack(fpn_features), dim=0)
        
        # SPIN Pyramid for multi-scale reasoning
        spin_out = self.spin_pyramid(combined_features)

        # Final segmentation output
        output = self.segmentation_head(spin_out)
        
        # Upsample output to match input size
        output = F.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=True)

        return output