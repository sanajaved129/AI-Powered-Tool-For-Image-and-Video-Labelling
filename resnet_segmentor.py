import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights

# Hard Tanh–Softplus Activation
class HardTanhSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.hardtanh = nn.Hardtanh(0, 1)

    def forward(self, x):
        return self.hardtanh(self.softplus(x))

hard_tanh_softplus = HardTanhSoftplus()

# Decoder block for U-Net style skip connections
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# ResNet Encoder + U-Net Style Decoder
class ResNetSegmentationModel(nn.Module):
    def __init__(self, num_classes=1, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Keep encoder layers accessible for skip connections
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)  # last skip now comes from conv1_out

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Early conv block
        conv1_out = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x)))  # (B, 64, 128, 128)
        x0 = self.encoder_maxpool(conv1_out)                                    # (B, 64, 64, 64)

        # Encoder forward pass with skip connections
        x1 = self.encoder_layer1(x0)    # (B, 64, 64, 64)
        x2 = self.encoder_layer2(x1)    # (B, 128, 32, 32)
        x3 = self.encoder_layer3(x2)    # (B, 256, 16, 16)
        x4 = self.encoder_layer4(x3)    # (B, 512, 8, 8)

        if self.debug_shapes:
            print(f"[DEBUG] Encoder: conv1_out={conv1_out.shape}, x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}")

        # Decoder with skip connections
        d4 = self.dec4(x4, x3)          # (B, 256, 16, 16)
        d3 = self.dec3(d4, x2)          # (B, 128, 32, 32)
        d2 = self.dec2(d3, x1)          # (B, 64, 64, 64)
        d1 = self.dec1(d2, conv1_out)   # (B, 32, 128, 128) ← now matches

        if self.debug_shapes:
            print(f"[DEBUG] Decoder: d4={d4.shape}, d3={d3.shape}, d2={d2.shape}, d1={d1.shape}")

        # Upsample to final size
        out = F.interpolate(d1, size=(256, 256), mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        out = hard_tanh_softplus(out)
        return out
