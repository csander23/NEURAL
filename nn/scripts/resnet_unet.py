"""ResNet18-encoder U-Net for dense cell segmentation.

Architecture
------------
Encoder  : ResNet18 (optionally pretrained on ImageNet) extracts 5 feature
           maps at decreasing spatial resolutions.
Decoder  : 4 transpose-conv + skip-concat + double-conv stages bring the
           feature map back to full resolution, plus a final 2× upsample.
Output   : (B, out_ch, H, W) raw logits — sigmoid / tanh are applied in the
           loss functions, not here.

Skip-connection spatial sizes for a 512×512 input
  x0 : (B,  64, 256, 256) – after conv1 + BN + ReLU  (stride 2)
  x1 : (B,  64, 128, 128) – after maxpool + layer1    (stride 4)
  x2 : (B, 128,  64,  64) – after layer2              (stride 8)
  x3 : (B, 256,  32,  32) – after layer3              (stride 16)
  x4 : (B, 512,  16,  16) – after layer4 (bottleneck) (stride 32)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetUNet(nn.Module):
    """ResNet18-encoder U-Net.

    Parameters
    ----------
    out_ch    : number of output channels (approach-specific)
    pretrained: load ImageNet weights for the encoder backbone
    """

    def __init__(
        self,
        out_ch: int = 2,
        pretrained: bool = True,
        apply_imagenet_norm: bool = False,
    ) -> None:
        """ResNet18 U-Net.

        apply_imagenet_norm: if True, applies the ImageNet (mean, std) transform
        to the input inside `forward` before the encoder sees it. Use when the
        encoder is pretrained on ImageNet and you want the conv1 filters to
        receive the same statistical distribution they were trained on.

        Subtlety re: z-scoring upstream — `load_native` already z-scores each
        summary channel to mean=0, std=1 per image, which is approximately
        where ImageNet-normalised inputs sit. Applying ImageNet norm ON TOP of
        z-scoring is therefore an *additional* affine that shifts/rescales the
        distribution further (per-channel mean lands around -2, std around 4-5)
        rather than landing it ON the ImageNet input distribution. Empirically
        a common practice when using a pretrained backbone, but treat as an
        ablate-able lever, not a fact.
        """
        super().__init__()
        self.apply_imagenet_norm = apply_imagenet_norm
        # Non-persistent buffers — don't appear in state_dict, so old checkpoints
        # without these still load under strict=True.
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # ── Encoder (ResNet18) ──────────────────────────────────────────────
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        rn = resnet18(weights=weights)

        self.enc_conv1   = rn.conv1    # (3, H, W) → (64, H/2, W/2)
        self.enc_bn1     = rn.bn1
        self.enc_relu    = rn.relu
        self.enc_maxpool = rn.maxpool  # → (64, H/4, W/4)
        self.enc_layer1  = rn.layer1   # → (64,  H/4,  W/4)
        self.enc_layer2  = rn.layer2   # → (128, H/8,  W/8)
        self.enc_layer3  = rn.layer3   # → (256, H/16, W/16)
        self.enc_layer4  = rn.layer4   # → (512, H/32, W/32)

        # ── Decoder ────────────────────────────────────────────────────────
        # level 4: H/32 → H/16, merge with x3 (256 ch)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _DoubleConv(256 + 256, 256)

        # level 3: H/16 → H/8, merge with x2 (128 ch)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _DoubleConv(128 + 128, 128)

        # level 2: H/8 → H/4, merge with x1 (64 ch)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _DoubleConv(64 + 64, 64)

        # level 1: H/4 → H/2, merge with x0 (64 ch)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _DoubleConv(32 + 64, 32)

        # level 0: H/2 → H, no skip
        self.up0  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec0 = _DoubleConv(16, 16)

        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_imagenet_norm:
            x = (x - self._imagenet_mean) / self._imagenet_std
        # ── Encode ──────────────────────────────────────────────────────────
        x0 = self.enc_relu(self.enc_bn1(self.enc_conv1(x)))   # (B, 64, 256, 256)
        x1 = self.enc_layer1(self.enc_maxpool(x0))            # (B, 64, 128, 128)
        x2 = self.enc_layer2(x1)                              # (B,128,  64,  64)
        x3 = self.enc_layer3(x2)                              # (B,256,  32,  32)
        x4 = self.enc_layer4(x3)                              # (B,512,  16,  16)

        # ── Decode ──────────────────────────────────────────────────────────
        d = self._up_cat_conv(self.up4, self.dec4, x4, x3)   # (B,256,32,32)
        d = self._up_cat_conv(self.up3, self.dec3, d,  x2)   # (B,128,64,64)
        d = self._up_cat_conv(self.up2, self.dec2, d,  x1)   # (B, 64,128,128)
        d = self._up_cat_conv(self.up1, self.dec1, d,  x0)   # (B, 32,256,256)
        d = self.dec0(self.up0(d))                            # (B, 16,512,512)

        return self.head(d)   # (B, out_ch, 512, 512)

    @staticmethod
    def _up_cat_conv(
        up: nn.Module,
        conv: nn.Module,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return conv(torch.cat([skip, x], dim=1))
