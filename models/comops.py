import collections.abc
import math
import torch
import torch.nn.functional as F
from torch import nn,Tensor
from itertools import repeat

from typing import Type, Callable, Tuple, Optional, Set, List, Union
from functools import partial

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B,C,H,W = x.size()
    # reshape
    x = x.view(B, groups, -1, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(B, -1, H, W)
    return x


class Permute(torch.nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    # 一个点跟其他所有点的行相对位置
    relative_coords[:, :, 0] += win_h - 1
    # 一个点跟其他所有点的列相对位置
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

class RelativeSelfAttention(nn.Module):
    def __init__(self, in_chs, num_groups=1, num_heads=1, block_size: Tuple[int, int] = (7, 7), attn_drop: float = 0., drop: float = 0., proj=True):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        qkdim = math.ceil(in_chs/num_groups/num_heads)*num_heads
        self.scale = qkdim//num_heads
        self.scale = self.scale**(-0.5)
        if self.block_size is not None:
            self.attn_area = block_size[0] * block_size[1]
            # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * block_size[0] - 1) * (2 * block_size[1] - 1), num_heads))
            # Get pair-wise relative position index for each token inside the window
            self.register_buffer("relative_position_index", get_relative_position_index(block_size[0],block_size[1]))
            # Init relative positional bias
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        # Init layers
        self.qk_proj =  nn.Linear(in_features=in_chs, out_features=2 * qkdim, bias=True)
        self.v_proj = nn.Linear(in_features=in_chs, out_features=in_chs, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = None
        if proj:
            self.proj = nn.Linear(in_features=in_chs, out_features=in_chs, bias=True)
            self.proj_drop = nn.Dropout(p=drop)

    def forward(self, x):
        B_, N, C = x.shape
        q, k = self.qk_proj(x).view(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        v = self.v_proj(x).view(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.block_size is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
            attn = attn+relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = self.attn_drop(attn)
        attn = self.softmax(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        if self.proj is not None:
            output = self.proj_drop(self.proj(output))
        return output


if __name__ == '__main__':
    pass