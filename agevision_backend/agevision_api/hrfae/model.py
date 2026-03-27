"""
HRFAE Model Architecture — Official Implementation
=====================================================
Exact reimplementation of HRFAE (High Resolution Face Age Editing)
matching the official InterDigitalInc/HRFAE repository.

Paper : "High Resolution Face Age Editing" (Xu Yao et al., 2020)
Repo  : https://github.com/InterDigitalInc/HRFAE

Architecture:
  Encoder  – 3→32→64→128 with 4 ResBlocks, skip connections
  Decoder  – Multiplicative age conditioning + skip concat + upsample
  Mod_Net  – One-hot age (101 classes) → Linear(101,128) → Sigmoid

Key design choices (from the paper):
  • Age conditioning via element-wise multiplication (NOT AdaIN)
  • U-Net-style skip connections between encoder and decoder
  • Spectral normalization on all conv layers
  • InstanceNorm2d with affine=True
  • LeakyReLU(0.2) throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ═══════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════

class Conv2d(nn.Module):
    """Convolution block: Padding → Conv → Norm → Activation.
    Matches the official HRFAE Conv2d class exactly."""

    def __init__(self, input_size, output_size, kernel_size, stride,
                 conv='conv', pad='mirror', norm='in', activ='relu',
                 sn=False):
        super(Conv2d, self).__init__()

        # ── Padding ──
        if pad == 'mirror':
            self.padding = nn.ReflectionPad2d(kernel_size // 2)
        elif pad == 'none':
            self.padding = None
        else:
            self.padding = nn.ReflectionPad2d(pad)

        # ── Convolution ──
        if conv == 'conv':
            self.conv = nn.Conv2d(input_size, output_size,
                                  kernel_size=kernel_size, stride=stride)

        # ── Normalization ──
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_size, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(output_size)
        elif norm == 'none':
            self.norm = None

        # ── Activation ──
        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)
        elif activ == 'none':
            self.relu = None

        # ── Spectral Normalization ──
        if sn:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        if self.padding:
            out = self.padding(x)
        else:
            out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.relu:
            out = self.relu(out)
        return out


class ResBlock(nn.Module):
    """Residual block with two Conv2d sub-blocks."""

    def __init__(self, input_size, kernel_size, stride,
                 conv='conv', pad='mirror', norm='in', activ='relu',
                 sn=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2d(input_size, input_size, kernel_size=kernel_size,
                   stride=stride, conv=conv, pad=pad, norm=norm,
                   activ=activ, sn=sn),
            Conv2d(input_size, input_size, kernel_size=kernel_size,
                   stride=stride, conv=conv, pad=pad, norm=norm,
                   activ=activ, sn=sn),
        )

    def forward(self, x):
        return x + self.block(x)


# ═══════════════════════════════════════════════════════════════
#  Encoder
# ═══════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Downsamples 256×256×3  →  64×64×128  with two skip outputs.

    Architecture:
      conv_1 : Conv(3→32,  k9, s1)  → 256×256×32
      conv_2 : Conv(32→64, k3, s2)  → 128×128×64   ← skip_2
      conv_3 : Conv(64→128,k3, s2)  →  64× 64×128  ← skip_1
      4× ResBlock(128)              →  64× 64×128   ← out

    Returns (out, skip_1=out_3, skip_2=out_2)
    """

    def __init__(self, input_size=3, activ='leakyrelu'):
        super(Encoder, self).__init__()
        self.conv_1 = Conv2d(input_size, 32, kernel_size=9, stride=1,
                             activ=activ, sn=True)
        self.conv_2 = Conv2d(32, 64, kernel_size=3, stride=2,
                             activ=activ, sn=True)
        self.conv_3 = Conv2d(64, 128, kernel_size=3, stride=2,
                             activ=activ, sn=True)
        self.res_block = nn.Sequential(
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
        )

    def forward(self, x):
        out_1 = self.conv_1(x)       # 256×256×32
        out_2 = self.conv_2(out_1)    # 128×128×64
        out_3 = self.conv_3(out_2)    #  64× 64×128
        out = self.res_block(out_3)   #  64× 64×128
        return out, out_3, out_2


# ═══════════════════════════════════════════════════════════════
#  Decoder
# ═══════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    Upsamples 64×64×128  →  256×256×3  using age-conditioned features
    and skip connections from the encoder.

    Age conditioning: element-wise multiplication of features with
    the 128-dim age vector (broadcast across spatial dims).

    Architecture:
      age_vec × out                         →  64× 64×128
      cat(out, skip_1)                      →  64× 64×256
      conv_1 : Upsample(2) + Conv(256→64)  → 128×128×64
      cat(out, skip_2)                      → 128×128×128
      conv_2 : Upsample(2) + Conv(128→32)  → 256×256×32
      conv_3 : Pad(4) + Conv(32→3, k9)     → 256×256×3
    """

    def __init__(self, output_size=3, activ='leakyrelu'):
        super(Decoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(256, 64, kernel_size=3, stride=1, activ=activ, sn=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(128, 32, kernel_size=3, stride=1, activ=activ, sn=True),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, output_size, kernel_size=9, stride=1),
        )

    def forward(self, x, age_vec, skip_1, skip_2):
        b, c = age_vec.size()
        age_vec = age_vec.view(b, c, 1, 1)
        out = age_vec * x                       # Multiplicative modulation
        out = torch.cat((out, skip_1), 1)        # 128 + 128 = 256 ch
        out = self.conv_1(out)                   # → 64 ch @ 128×128
        out = torch.cat((out, skip_2), 1)        # 64 + 64 = 128 ch
        out = self.conv_2(out)                   # → 32 ch @ 256×256
        out = self.conv_3(out)                   # → 3 ch  @ 256×256
        return out


# ═══════════════════════════════════════════════════════════════
#  Age Modulation Network
# ═══════════════════════════════════════════════════════════════

class Mod_Net(nn.Module):
    """
    Maps an integer age (0–100) to a 128-dim modulation vector
    via one-hot encoding and a single linear layer.

    Forward: age_int [B] → one_hot [B,101] → Linear(101,128) → Sigmoid → [B,128]
    """

    def __init__(self):
        super(Mod_Net, self).__init__()
        self.fc_mix = nn.Linear(101, 128, bias=False)

    def forward(self, x):
        b_s = x.size(0)
        z = torch.zeros(b_s, 101, device=x.device, dtype=torch.float32)
        for i in range(b_s):
            z[i, x[i].long()] = 1
        y = self.fc_mix(z)
        y = torch.sigmoid(y)
        return y


# ═══════════════════════════════════════════════════════════════
#  PatchGAN Discriminator  (training only)
# ═══════════════════════════════════════════════════════════════

class Dis_PatchGAN(nn.Module):
    """PatchGAN discriminator with spectral normalization."""

    def __init__(self, input_size=3):
        super(Dis_PatchGAN, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(input_size, 32, kernel_size=4, stride=2,
                   norm='none', activ='leakyrelu', sn=True),
            Conv2d(32, 64, kernel_size=4, stride=2,
                   norm='batch', activ='leakyrelu', sn=True),
            Conv2d(64, 128, kernel_size=4, stride=2,
                   norm='batch', activ='leakyrelu', sn=True),
            Conv2d(128, 256, kernel_size=4, stride=2,
                   norm='batch', activ='leakyrelu', sn=True),
            Conv2d(256, 512, kernel_size=4, stride=1,
                   norm='batch', activ='leakyrelu', sn=True),
            Conv2d(512, 1, kernel_size=4, stride=1,
                   norm='none', activ='none', sn=True),
        )

    def forward(self, x):
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════
#  VGG Age Classifier  (for perceptual/age loss during training)
# ═══════════════════════════════════════════════════════════════

class VGG(nn.Module):
    """VGG-16 with 101-class age output (DEX architecture).
    Used for perceptual loss and age classification during training.
    
    Pretrained weights from:
      https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
    """

    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(25088, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8_101 = nn.Linear(4096, 101, bias=True)

        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['p3'] = self.pool3(out['r33'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['p4'] = self.pool4(out['r43'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['p5'] = self.pool5(out['r53'])
        out['p5'] = out['p5'].view(out['p5'].size(0), -1)
        out['fc6'] = F.relu(self.fc6(out['p5']))
        out['fc7'] = F.relu(self.fc7(out['fc6']))
        out['fc8'] = self.fc8_101(out['fc7'])
        return out


# ═══════════════════════════════════════════════════════════════
#  Complete HRFAE Model  (bundles Encoder + Decoder + Mod_Net)
# ═══════════════════════════════════════════════════════════════

class HRFAE(nn.Module):
    """Complete HRFAE model for inference.

    Usage:
        model = HRFAE()
        # x: [B, 3, 256, 256] face image in [0, 1] range
        # target_age: [B] integer ages (0–100)
        output = model(x, target_age)
    """

    def __init__(self):
        super(HRFAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mod_net = Mod_Net()

    def forward(self, x, target_age):
        """
        Args:
            x           : [B, 3, 256, 256] input face image in [0, 1]
            target_age  : [B] integer target ages (0–100)
        Returns:
            output      : [B, 3, 256, 256] aged face image
        """
        content, skip_1, skip_2 = self.encoder(x)
        age_vec = self.mod_net(target_age)
        output = self.decoder(content, age_vec, skip_1, skip_2)
        return output

    def get_content(self, x):
        """Extract content/identity features (for identity loss)."""
        content, _, _ = self.encoder(x)
        return content

    def load_checkpoint(self, checkpoint_path, device='cpu'):
        """Load pre-trained weights from an official HRFAE checkpoint.

        Supports multiple checkpoint formats:
          1. Official HRFAE format: {'enc_state_dict', 'dec_state_dict', 'mlp_style_state_dict'}
          2. Short-key format:     {'enc', 'dec', 'mod'}
          3. Our training format:  {'encoder', 'decoder', 'mod_net'}
          4. Bundled format:       {'model'}
          5. Raw state_dict
        """
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)

        if 'enc_state_dict' in ckpt:
            # Official HRFAE checkpoint (from InterDigitalInc/HRFAE)
            self.encoder.load_state_dict(ckpt['enc_state_dict'])
            self.decoder.load_state_dict(ckpt['dec_state_dict'])
            self.mod_net.load_state_dict(ckpt['mlp_style_state_dict'])
        elif 'enc' in ckpt:
            # Short-key format
            self.encoder.load_state_dict(ckpt['enc'])
            self.decoder.load_state_dict(ckpt['dec'])
            self.mod_net.load_state_dict(ckpt['mod'])
        elif 'encoder' in ckpt:
            # Our training script format
            self.encoder.load_state_dict(ckpt['encoder'])
            self.decoder.load_state_dict(ckpt['decoder'])
            self.mod_net.load_state_dict(ckpt['mod_net'])
        elif 'model' in ckpt:
            self.load_state_dict(ckpt['model'])
        else:
            # Try loading as raw state_dict
            self.load_state_dict(ckpt)

        epoch = ckpt.get('n_epoch', ckpt.get('epoch', '?'))
        print(f"[HRFAE] Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
