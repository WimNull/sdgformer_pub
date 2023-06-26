import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from typing import Type, Callable, Tuple, Optional, Set, List, Union
from einops import rearrange

from .comops import channel_shuffle
from .comops import DropPath
from .comops import to_2tuple


class LayerNormNCHW(nn.Module):
    def __init__(self, in_chs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_chs)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ShuffleChannel(nn.Module):
    def __init__(self, groups) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x):
        x = channel_shuffle(x, self.groups)
        return x


class SE(nn.Module):
    def __init__(self, in_chs, ratio=0.25, act_layer=nn.ReLU, gate_layer=nn.Sigmoid, bksz=(1,1)):
        super().__init__()
        rd_channels = round(in_chs*ratio)
        self.bksz = bksz
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()
    
    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, self.bksz)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        if self.bksz[0]!=1:
            x_se = F.interpolate(x_se, (x.shape[-2], x.shape[-1]))
        return x * self.gate(x_se)

class SdgAttn(nn.Module):
    def __init__(self, in_chs, num_heads=1, block_size=7, in_rel=(56, 56), drop=0., drop_attn=0., norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU) -> None:
        super().__init__()
        self.block_size = to_2tuple(block_size)
        half_chs = in_chs//2
        self.activation = act_layer()
        ks = min(7, in_rel[0]//2)
        self.conv_local = nn.Sequential(
            nn.Conv2d(half_chs, half_chs, ks, 1, padding=ks//2, groups=half_chs),
            norm_layer(half_chs),
            nn.Conv2d(half_chs, in_chs, 1, 1),
            act_layer(),
            nn.Conv2d(in_chs, half_chs, 1, 1),
        )
        self.qk = nn.Sequential(
            nn.Conv2d(half_chs, half_chs, 1, 1),
            # norm_layer(half_chs),
            # act_layer(),
        )
        self.attn_global = nn.Conv2d(block_size*block_size, block_size*block_size, 1, 1)
        self.proj = nn.Conv2d(half_chs, half_chs, 1, 1)
        self.se = SE(in_chs, act_layer=act_layer)
    def forward(self, x):
        B,C,H,W = x.shape
        bksz = self.block_size
        x1,x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv_local(x1)
        x2 = self.qk(x2)
        gx2 = rearrange(x2, 'b c (h p1) (w p2) -> (b c) (h w) p1 p2', h=bksz[0], w=bksz[1])
        gx2 = self.attn_global(self.activation(gx2))
        gx2 = rearrange(gx2, '(b c) (h w) p1 p2 -> b c (h p1) (w p2)', b=B, h=bksz[0], w=bksz[1])
        x2 = x2*torch.sigmoid(gx2)
        # x2 = x2+torch.clamp(gx2, -6, 6)
        x2 = self.proj(x2)
        x = torch.cat([x1,x2], dim=1)
        x = self.se(x)
        return x



class SdgBlock(nn.Module):
    def __init__(self, in_chs, num_heads=1, in_rel=(56, 56), block_size=7, sidx=0, bidx=0, 
        drop: float = 0., drop_path=0., drop_attn: float = 0., mlp_ratio:int=4, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, 
    ):
        super().__init__()
        hidden_features = mlp_ratio*in_chs
        hidden_features = round(hidden_features/num_heads)*num_heads
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.activation = act_layer()
        self.norm1 = norm_layer(in_chs) if bidx!=0 else nn.Identity()
        self.norm1 = norm_layer(in_chs)
        self.attn = SdgAttn(in_chs, num_heads, block_size, in_rel, drop, drop_attn, norm_layer=norm_layer, act_layer=act_layer)
        self.norm2 = norm_layer(in_chs)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_chs, hidden_features, 1, 1, groups=num_heads),
            ShuffleChannel(num_heads),
            act_layer(),
            nn.Conv2d(hidden_features, in_chs, 1, 1, groups=num_heads),
        )

    def forward(self, x):
        
        # x = x+self.drop_path(self.attn(self.norm1(x)))
        # x = x+self.drop_path(self.mlp(self.norm2(x)))
        # x = x+self.drop_path(self.norm1(self.attn(x)))
        # x = x+self.drop_path(self.norm2(self.mlp(x)))

        # x = self.activation(x+self.drop_path(self.norm1(self.attn(x))))
        # x = self.activation(x+self.drop_path(self.norm2(self.mlp(x))))

        x = self.activation(x) + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm1(x)
        # # x = x + self.drop_path(self.attn(self.activation(x)))
        # x = self.activation(x) + self.drop_path(self.mlp(self.norm2(x)))
        # x = self.norm1(x)
        
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_rel, in_chs, out_chs, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = in_chs
        self.reduction = nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2)
        self.norm = norm_layer(out_chs)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SdgStage(nn.Module):
    def __init__(self, depth, in_chs, out_chs, num_heads=1, in_rel=(56, 56), block_size=7,
            drop_attn: float = 0., drop: float = 0., drop_path:List[float] = 0., mlp_ratio:int=4,
            downsample=None, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, sidx=0,
        ):
        super().__init__()
        # build blocks
        self.blocks = nn.Sequential(*[
            SdgBlock(in_chs=in_chs, in_rel=in_rel, num_heads=num_heads, block_size=block_size, norm_layer=norm_layer, act_layer=act_layer, 
                            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i], drop_attn=drop_attn, sidx=sidx, bidx=i, )
            for i in range(depth)])

        # patch merging layer
        self.downsample = nn.Identity() if downsample is None else downsample(in_rel, in_chs=in_chs, out_chs=out_chs)

    def forward(self, x):
        x = self.blocks(x)
        x = self.downsample(x)
        return x


# channels=(64, 128, 256, 512)
class SdgFormer(nn.Module):
    def __init__(self, in_chs=3, in_rel=(224, 224), depths=(2,2,6,2), num_heads=(4, 4, 4, 4), channels=[80, 160, 320, 640], num_classes=1000, 
        patch_size=4, block_size=7, drop_attn: float = 0., drop=0., drop_path=0.1, mlp_ratio=4, norm_layer = nn.BatchNorm2d, act_layer=nn.ReLU, 
    ):
        super().__init__()
        self.num_stages = len(depths)
        in_rel = (in_rel[0]//patch_size, in_rel[1]//patch_size)
        channels = channels+[channels[-1]]
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stem = nn.Sequential(
            # nn.Conv2d(in_chs, channels[0], kernel_size=5, stride=patch_size, padding=2, bias=False),
            nn.Conv2d(in_channels=in_chs, out_channels=channels[0], kernel_size=patch_size, stride=patch_size),
            norm_layer(channels[0]),
        )

        self.stages = []
        for i_layer in range(self.num_stages):
            layer = SdgStage(in_chs=channels[i_layer], out_chs=channels[i_layer+1], num_heads=num_heads[i_layer], 
                depth=depths[i_layer], in_rel=(in_rel[0]//2**i_layer, in_rel[1]//2**i_layer), block_size=block_size, 
                downsample=PatchMerging if (i_layer < self.num_stages - 1) else None, norm_layer=norm_layer, act_layer=act_layer, 
                drop=drop, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], drop_attn=drop_attn, mlp_ratio=mlp_ratio, sidx=i_layer, 
            )
            self.stages.append(layer)
        self.stages = nn.Sequential(*self.stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

def sdgformer_tiny(num_classes=1000):
    return SdgFormer(num_classes=num_classes, act_layer=partial(nn.GELU), mlp_ratio=2)

def sdgformer_small(num_classes=1000):
    return SdgFormer(num_classes=num_classes, depths=(2,4,8,2), num_heads=(2, 2, 4, 4), act_layer=partial(nn.GELU), mlp_ratio=4)

def sdgformer_base(num_classes=1000):
    return SdgFormer(num_classes=num_classes, depths=(2,6,14,2), num_heads=(2, 2, 2, 2), channels=[96, 192, 384, 768], act_layer=partial(nn.GELU), mlp_ratio=4)


if __name__ == '__main__':
    torch.set_printoptions(precision=2, threshold=1000, linewidth=1000, sci_mode=False)
    def testmodel():
        x = torch.rand((2, 3, 224, 224))
        model = SdgFormer()
        y = model(x)
    testmodel()

