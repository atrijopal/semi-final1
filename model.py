"""
Standalone model definition for the submitted checkpoint -- "Shipped"
(plain NAFNet-full architecture, bicubic-first trunk, PixelShuffle
decoder), 29.07M params, additive-residual joint denoise + 2x
super-resolution.

Deliberately self-contained (torch only, no external repo/package
dependency) so run.py has zero non-trivial setup. The NAFBlock /
LayerNorm2d building blocks are extracted, unmodified, from the official
megvii-research/NAFNet repository (Chen et al., "Simple Baselines for
Image Restoration", arXiv:2204.04676).

Architecture, in brief: the 128x128 noisy low-res input is first
bicubic-upsampled to 256x256 -- that upsampled image is both the trunk's
only input and the base the network adds a learned residual correction
onto. The trunk itself is a 4-level NAFNet encoder/decoder (bias-free
convolutions, channel width 32, block counts 2-2-4-8 / 12 middle / 2-2-2-2)
with skip connections and 4x PixelShuffle(2) upsampling in the decoder.
Final output = clamp(bicubic + correction, 0, 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- extracted, unmodified from megvii-research/NAFNet ---------------------

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


# ---- this project's model: bicubic-first trunk, residual-add head ----------

def strip_conv_bias(module: nn.Module) -> nn.Module:
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            m.bias = None
    return module


class NAFNetRestoration(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)
        self.head_add = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        strip_conv_bias(self)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, bicubic, nlr=None):
        """bicubic: (N,1,H,W) bicubic-upsampled input -- both the trunk's
        only input and the residual-add baseline. nlr is accepted for
        interface compatibility with other models in this project but is
        unused here (the shipped architecture never sees the native-res
        input directly)."""
        _, _, H, W = bicubic.shape
        x = self.check_image_size(bicubic)
        x = self.intro(x)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        feat = x[:, :, :H, :W]
        R = self.head_add(feat)
        return torch.clamp(bicubic + R, 0.0, 1.0)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


FULLSIZE_CONFIG = dict(width=32, enc_blk_nums=(2, 2, 4, 8), middle_blk_num=12, dec_blk_nums=(2, 2, 2, 2))


def build_model(**kwargs):
    cfg = dict(FULLSIZE_CONFIG)
    cfg.update(kwargs)
    return NAFNetRestoration(img_channel=1, **cfg)
