# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Dict, Any, List, Set, Optional, Union, Callable, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
import lightning as L

from src.model.blocks import *
from src.model.transformation import *


WO_SELF_ATT = False # without self attention
_NUM_CROSS_ATT = -1
ndims = 3 # H,W,D

class Attention(nn.Module):
    """
    Attention module for hierarchical vision transformer.

    This module implements both local and global attention mechanisms.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: Union[int, List[int]],
        attention_type: str = "local",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ) -> None:
        """
        Initialize the Attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            patch_size (Union[int, List[int]]): Size of the patches.
            attention_type (str): Type of attention mechanism ("local" or "global").
            qkv_bias (bool): Whether to use bias in query, key, value projections.
            qk_scale (Optional[float]): Scale factor for query-key dot product.
            attn_drop (float): Dropout rate for attention matrix.
            proj_drop (float): Dropout rate for output projection.
        """
        super().__init__()

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.patch_size: List[int] = [patch_size] * ndims if isinstance(patch_size, int) else patch_size
        self.attention_type: str = attention_type
        
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.head_dim: int = dim // num_heads
        self.scale: float = qk_scale or self.head_dim ** -0.5

        # Skip initialization if using local attention without self-attention
        if self.attention_type == "local" and WO_SELF_ATT:
            return

        # Initialize query, key, value projections based on attention type
        if attention_type == "local":
            self.qkv: nn.Linear = nn.Linear(dim, dim * 3, bias=qkv_bias)
        elif attention_type == "global":
            self.qkv: nn.Linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop: nn.Dropout = nn.Dropout(attn_drop)
        self.proj: nn.Linear = nn.Linear(dim, dim)
        self.proj_drop: nn.Dropout = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, q_ms: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (Tensor): Input tensor.
            q_ms (Optional[Tensor]): Query tensor for global attention.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        B_, N, C = x.size()

        # Return input if using local attention without self-attention
        if self.attention_type == "local" and WO_SELF_ATT:
            return x

        if self.attention_type == "local":
            qkv: Tensor = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
        else:
            B: int = q_ms.size()[0]
            kv: Tensor = self.qkv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            q: Tensor = self._process_global_query(q_ms, B, B_, N, C)

        # Compute attention scores and apply attention
        attn: Tensor = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x: Tensor = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def _process_global_query(self, q_ms: Tensor, B: int, B_: int, N: int, C: int) -> Tensor:
        """
        Process the global query tensor.

        Args:
            q_ms (Tensor): Global query tensor.
            B (int): Batch size of q_ms.
            B_ (int): Batch size of input tensor.
            N (int): Number of patches.
            C (int): Channel dimension.

        Returns:
            Tensor: Processed global query tensor.
        """
        q_tmp: Tensor = q_ms.reshape(B, self.num_heads, N, C // self.num_heads)
        div_, rem_ = divmod(B_, B)
        q_tmp = q_tmp.repeat(div_, 1, 1, 1)
        q_tmp = q_tmp.reshape(B * div_, self.num_heads, N, C // self.num_heads)
        
        q: Tensor = torch.zeros(B_, self.num_heads, N, C // self.num_heads, device=q_ms.device)
        q[:B*div_] = q_tmp
        if rem_ > 0:
            q[B*div_:] = q_tmp[:rem_]
        
        return q * self.scale


def get_patches(x: Tensor, patch_size: int) -> Tuple[Tensor, int, int, int]:
    """
    Divide the input tensor into patches and reshape them for processing.

    Args:
        x (Tensor): Input tensor of shape (B, H, W, D, C).
        patch_size (int): Size of each patch.

    Returns:
        Tuple[Tensor, int, int, int]: A tuple containing:
            - windows: Reshaped tensor of patches.
            - H, W, D: Updated dimensions of the input tensor.
    """
    B, H, W, D, C = x.size()
    nh: float = H / patch_size
    nw: float = W / patch_size
    nd: float = D / patch_size

    # Check if downsampling is required
    down_req: float = (nh - int(nh)) + (nw - int(nw)) + (nd - int(nd))
    if down_req > 0:
        # Downsample the input tensor to fit patch size
        new_dims: List[int] = [int(nh) * patch_size, int(nw) * patch_size, int(nd) * patch_size]
        x = downsampler_fn(x.permute(0, 4, 1, 2, 3), new_dims).permute(0, 2, 3, 4, 1)
        B, H, W, D, C = x.size()

    # Reshape the tensor into patches
    x = x.view(B, H // patch_size, patch_size,
               W // patch_size, patch_size,
               D // patch_size, patch_size,
               C)
    
    # Rearrange dimensions and flatten patches
    windows: Tensor = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, patch_size, patch_size, patch_size, C)
    
    return windows, H, W, D


def get_image(windows: Tensor, patch_size: int, Hatt: int, Watt: int, Datt: int, H: int, W: int, D: int) -> Tensor:
    """
    Reconstruct the image from windows (patches).

    Args:
        windows (Tensor): Input tensor containing the windows.
        patch_size (int): Size of each patch.
        Hatt, Watt, Datt (int): Dimensions of the attention space.
        H, W, D (int): Original dimensions of the image.

    Returns:
        Tensor: Reconstructed image.
    """
    # Calculate batch size
    B: int = int(windows.size(0) / ((Hatt * Watt * Datt) // (patch_size ** 3)))
    
    # Reshape windows into image
    x: Tensor = windows.view(B, 
                    Hatt // patch_size,
                    Watt // patch_size,
                    Datt // patch_size,
                    patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hatt, Watt, Datt, -1)

    # Downsample if necessary
    if H != Hatt or W != Watt or D != Datt:
        x = downsampler_fn(x.permute(0, 4, 1, 2, 3), [H, W, D]).permute(0, 2, 3, 4, 1)
    return x

class ViTBlock(nn.Module):
    """
    Vision Transformer Block.
    """
    def __init__(self,
                 embed_dim: int,
                 input_dims: List[int],
                 num_heads: int,
                 mlp_type: str,
                 patch_size: int,
                 mlp_ratio: float,
                 qkv_bias: bool,
                 qk_scale: Optional[float],
                 drop: float,
                 attn_drop: float,
                 drop_path: float,
                 act_layer: str,
                 attention_type: str,
                 norm_layer: Callable[..., nn.Module],
                 layer_scale: Optional[float]):
        super().__init__()
        self.patch_size: int = patch_size
        self.num_windows: int = prod_func([d // patch_size for d in input_dims])

        self.norm1: nn.Module = norm_layer(embed_dim)
        self.attn: nn.Module = Attention(
            embed_dim,
            attention_type=attention_type,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path: nn.Module = timm_DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2: nn.Module = norm_layer(embed_dim)
        self.mlp: nn.Module = MLP(
            in_feats=embed_dim,
            hid_feats=int(embed_dim * mlp_ratio),
            act_name=act_layer,
            drop=drop,
            MLP_type=mlp_type
        )

        self.layer_scale: bool = layer_scale is not None and isinstance(layer_scale, (int, float))
        if self.layer_scale:
            self.gamma1: nn.Parameter = nn.Parameter(layer_scale * torch.ones(embed_dim), requires_grad=True)
            self.gamma2: nn.Parameter = nn.Parameter(layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.gamma1: float = 1.0
            self.gamma2: float = 1.0

    def forward(self, x: Tensor, q_ms: Optional[Tensor]) -> Tensor:
        B, H, W, D, C = x.size()
        shortcut: Tensor = x

        x = self.norm1(x)
        x_windows, Hatt, Watt, Datt = get_patches(x, self.patch_size)
        x_windows = x_windows.view(-1, self.patch_size ** 3, C)

        attn_windows: Tensor = self.attn(x_windows, q_ms)
        x = get_image(attn_windows, self.patch_size, Hatt, Watt, Datt, H, W, D)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding layer.
    """
    def __init__(self, in_chans: int = 3, out_chans: int = 32,
                 drop_rate: float = 0,
                 kernel_size: int = 3,
                 stride: int = 1, padding: int = 1,
                 dilation: int = 1, groups: int = 1, bias: bool = False) -> None:
        super().__init__()

        Convnd: Type[nn.Module] = getattr(nn, f"Conv{ndims}d")
        self.proj: nn.Module = Convnd(in_channels=in_chans, out_channels=out_chans,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        self.drop: nn.Module = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.proj(x))
        return x

class ViTLayer(nn.Module):
    """
    Vision Transformer Layer.
    """
    def __init__(
        self,
        attention_type: str,
        dim: int,
        dim_out: int,
        depth: int,
        input_dims: List[int],
        num_heads: int,
        patch_size: int,
        mlp_type: str,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop: float,
        attn_drop: float,
        drop_path: Union[float, List[float]],
        norm_layer: Callable[..., nn.Module],
        norm_type: str,
        layer_scale: Optional[float],
        act_layer: str
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.embed_dim: int = dim
        self.input_dims: List[int] = input_dims
        self.blocks: nn.ModuleList = nn.ModuleList([
            ViTBlock(
                embed_dim=dim,
                num_heads=num_heads,
                mlp_type=mlp_type,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attention_type=attention_type,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[k] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
                input_dims=input_dims
            )
            for k in range(depth)
        ])

    def forward(self, inp: Tensor, q_ms: Optional[Tensor], CONCAT_ok: bool) -> Tensor:
        x: Tensor = inp.clone()
        x = rearrange(x, 'b c h w d -> b h w d c')
        
        if q_ms is not None:
            q_ms = rearrange(q_ms, 'b c h w d -> b h w d c')

        for blk in self.blocks:
            if q_ms is None:
                x = blk(x, None)
            else:
                q_ms_patches, _, _, _ = get_patches(q_ms, self.patch_size)
                q_ms_patches = q_ms_patches.view(-1, self.patch_size ** ndims, x.size()[-1])
                x = blk(x, q_ms_patches)

        x = rearrange(x, 'b h w d c -> b c h w d')

        if CONCAT_ok:
            x = torch.cat((inp, x), dim=-1)
        else:
            x = inp + x
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) module for hierarchical feature processing.
    """
    def __init__(self,
                 PYR_SCALES=None,
                 feats_num=None,
                 hid_dim=None,
                 depths=None,
                 patch_size=None,
                 mlp_ratio=None,
                 num_heads=None,
                 mlp_type=None,
                 norm_type=None,
                 act_layer=None,
                 drop_path_rate: float = 0.2,
                 qkv_bias: bool = True,
                 qk_scale: bool = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 img_size=None,
                 NUM_CROSS_ATT=-1):
        super().__init__()

        # Determine the number of levels for processing
        num_levels = len(feats_num)
        num_levels = min(num_levels, NUM_CROSS_ATT) if NUM_CROSS_ATT > 0 else num_levels
        if WO_SELF_ATT:
            num_levels -= 1

        # Ensure patch_size is a list
        patch_size = patch_size if isinstance(patch_size, list) else [patch_size for _ in range(num_levels)]
        hwd = img_size[-1]

        # Create patch embedding layers
        self.patch_embed = nn.ModuleList([
            PatchEmbed(
                in_chans=feats_num[i],
                out_chans=hid_dim,
                drop_rate=drop_rate
            ) for i in range(num_levels)
        ])

        # Generate drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Create ViT layers
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            level = ViTLayer(
                dim=hid_dim,
                dim_out=hid_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                patch_size=patch_size[i],
                mlp_type=mlp_type,
                attention_type="local" if i == 0 else "global",
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                input_dims=img_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
                norm_type=norm_type,
                act_layer=act_layer
            )
            self.levels.append(level)

    def _init_weights(self, m):
        """Initialize the weights of the module."""
        if isinstance(m, nn.Linear):
            timm_trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Return keywords for no weight decay."""
        return {'rpb'}

    def forward(self, KQs, CONCAT_ok: bool = False):
        """
        Forward pass of the ViT module.

        Args:
            KQs (List[Tensor]): List of input tensors for each level.
            CONCAT_ok (bool): Flag to determine if concatenation is allowed.

        Returns:
            Tensor: Processed output tensor.
        """
        for i, (patch_embed_, level) in enumerate(zip(self.patch_embed, self.levels)):
            if i == 0:
                # First level: process input without cross-attention
                Q = patch_embed_(KQs[i])
                x = level(Q, None, CONCAT_ok=CONCAT_ok)
                Q = patch_embed_(x)
            else:
                # Subsequent levels: process with cross-attention
                K = patch_embed_(KQs[i])
                x = level(Q, K, CONCAT_ok=CONCAT_ok)
                Q = x.clone()

        return x


class EncoderCnnBlock(nn.Module):
    """
    Convolutional block for the encoder part of the network.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=1,
        bias=False,
        affine=True,
        eps=1e-05
    ):
        super().__init__()

        # First convolutional block
        conv_block_1 = [
            nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        # Second convolutional block
        conv_block_2 = [
            nn.Conv3d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        # Combine both blocks
        self._block = nn.Sequential(
            *conv_block_1,
            *conv_block_2
        )

    def forward(self, x):
        """Forward pass of the EncoderCnnBlock."""
        return self._block(x)


class Decoder(nn.Module):
    """
    Decoder module for the hierarchical vision transformer.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._num_stages: int = config['num_stages']
        self.use_seg: bool = config['use_seg_loss']

        # Determine channels of encoder feature maps
        encoder_out_channels: torch.Tensor = torch.tensor([config['start_channels'] * 2**stage for stage in range(self._num_stages)])

        # Estimate required stages
        required_stages: Set[int] = set(int(fmap[-1]) for fmap in config['out_fmaps'])
        self._required_stages: Set[int] = required_stages

        earliest_required_stage: int = min(required_stages)

        # Lateral connections
        lateral_in_channels: torch.Tensor = encoder_out_channels[earliest_required_stage:]
        lateral_out_channels: torch.Tensor = lateral_in_channels.clip(max=config['fpn_channels'])

        self._lateral: nn.ModuleList = nn.ModuleList([
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
            for in_ch, out_ch in zip(lateral_in_channels, lateral_out_channels)
        ])
        self._lateral_levels: int = len(self._lateral)

        # Output layers
        out_in_channels: List[int] = [lateral_out_channels[-self._num_stages + required_stage].item() for required_stage in required_stages]
        out_out_channels: List[int] = [int(config['fpn_channels'])] * len(out_in_channels)
        out_out_channels[0] = int(config['fpn_channels'])

        self._out: nn.ModuleList = nn.ModuleList([
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
            for in_ch, out_ch in zip(out_in_channels, out_out_channels)
        ])

        # Upsampling layers
        self._up: nn.ModuleList = nn.ModuleList([
            nn.ConvTranspose3d(
                in_channels=list(reversed(lateral_out_channels))[level],
                out_channels=list(reversed(lateral_out_channels))[level+1],
                kernel_size=list(reversed(config['strides']))[level],
                stride=list(reversed(config['strides']))[level]
            )
            for level in range(len(lateral_out_channels)-1)
        ])

        # Multi-scale attention
        self.hierarchical_dec: nn.ModuleList = self._create_hierarchical_layers(config, out_out_channels)

        if self.use_seg:
            self._seg_head: nn.ModuleList = nn.ModuleList([
                nn.Conv3d(out_ch, config['num_organs'] + 1, kernel_size=1, stride=1)
                for out_ch in out_out_channels
            ])

    def _create_hierarchical_layers(self, config: Dict[str, Any], out_out_channels: List[int]) -> nn.ModuleList:
        """Create hierarchical layers for multi-scale attention."""
        out: nn.ModuleList = nn.ModuleList()
        img_size: List[List[int]] = []
        feats_num: List[int] = []

        for k, out_ch in enumerate(out_out_channels):
            img_size.append([int(item/(2**(self._num_stages-k-1))) for item in config['data_size']])
            feats_num.append(out_ch)
            n: int = len(feats_num)

            if k == 0:
                out.append(nn.Identity())
            else:
                out.append(
                    ViT(
                        NUM_CROSS_ATT=config.get('NUM_CROSS_ATT', _NUM_CROSS_ATT),
                        PYR_SCALES=[1.],
                        feats_num=feats_num,
                        hid_dim=int(config.get('fpn_channels', 64)),
                        depths=[int(config.get('depths', 1))]*n,
                        patch_size=[int(config.get('patch_size', 2))]*n,
                        mlp_ratio=int(config.get('mlp_ratio', 2)),
                        num_heads=[int(config.get('num_heads', 32))]*n,
                        mlp_type='basic',
                        norm_type='BatchNorm2d',
                        act_layer='gelu',
                        drop_path_rate=config.get('drop_path_rate', 0.2),
                        qkv_bias=config.get('qkv_bias', True),
                        qk_scale=None,
                        drop_rate=config.get('drop_rate', 0.),
                        attn_drop_rate=config.get('attn_drop_rate', 0.),
                        norm_layer=nn.LayerNorm,
                        layer_scale=1e-5,
                        img_size=img_size
                    )
                )
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass of the Decoder."""
        lateral_out: List[Tensor] = [lateral(fmap) for lateral, fmap in zip(self._lateral, list(x.values())[-self._lateral_levels:])]

        up_out: List[Tensor] = []
        for idx, x in enumerate(reversed(lateral_out)):
            if idx != 0:
                x = x + up

            if idx < self._lateral_levels - 1:
                up = self._up[idx](x)

            up_out.append(x)

        cnn_outputs: Dict[int, Tensor] = {stage: self._out[idx](fmap) for idx, (fmap, stage) in enumerate(zip(reversed(up_out), self._required_stages))}
        return self._forward_hierarchical(cnn_outputs)

    def _forward_hierarchical(self, cnn_outputs: Dict[int, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through the hierarchical decoder."""
        xs: List[Tensor] = [cnn_outputs[key].clone() for key in range(max(cnn_outputs.keys()), min(cnn_outputs.keys())-1, -1)]

        out_dict: Dict[str, Tensor] = {}
        QK: List[Tensor] = []
        for i, key in enumerate(range(max(cnn_outputs.keys()), min(cnn_outputs.keys())-1, -1)):
            QK = [xs[i]] + QK
            if i == 0:
                Pi = QK[0]
            else:
                Pi = self.hierarchical_dec[i](QK)
            QK[0] = Pi
            out_dict[f'P{key}'] = Pi

            if self.use_seg:
                Pi_seg = self._seg_head[i](Pi)
                out_dict[f'S{key}'] = Pi_seg

        return out_dict




class HierarchicalViT(nn.Module):
    """
    Hierarchical Vision Transformer (HViT) for image processing tasks.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration parameters
        self.backbone: str = config['backbone_net']
        in_channels: int = 2 * config.get('in_channels', 1)  # source + target
        kernel_size: int = config.get('kernel_size', 3)
        emb_dim: int = config.get('start_channels', 32)
        data_size: Tuple[int, ...] = config.get('data_size', [160, 192, 224])
        self.out_fmaps: List[str] = config.get('out_fmaps', ['P4', 'P3', 'P2', 'P1'])

        # Calculate number of stages
        num_stages: int = min(int(math.log2(min(data_size))) - 1,
                              max(int(fmap[-1]) for fmap in self.out_fmaps) + 1)

        strides: List[int] = [1] + [2] * (num_stages - 1)
        kernel_sizes: List[int] = [kernel_size] * num_stages

        config['num_stages'] = num_stages
        config['strides'] = strides

        # Build encoder
        self._encoder: nn.ModuleList = nn.ModuleList()
        if self.backbone in ['fpn', 'FPN']:
            for k in range(num_stages):
                blk = EncoderCnnBlock(
                    in_channels=in_channels,
                    out_channels=emb_dim,
                    kernel_size=kernel_sizes[k],
                    stride=strides[k]
                )
                self._encoder.append(blk)

                in_channels = emb_dim
                emb_dim *= 2

        # Build decoder
        if self.backbone in ['fpn', 'FPN']:
            self._decoder: Decoder = Decoder(config)
    
    def init_weights(self) -> None:
        """
        Initialize model weights.
        """
        # TODO: Implement weight initialization

    def forward(self, x: Tensor, verbose: bool = False) -> Dict[str, Tensor]:
        """
        Forward pass of the HierarchicalViT model.

        Args:
            x (Tensor): Input tensor.
            verbose (bool): If True, print shape information.

        Returns:
            Dict[str, Tensor]: Output feature maps.
        """
        down: Dict[str, Tensor] = {}
        if self.backbone in ['fpn', 'FPN']:
            for stage_id, module in enumerate(self._encoder):
                x = module(x)
                down[f'C{stage_id}'] = x
            up = self._decoder(down)

        if verbose:
            for key, item in down.items():
                print(f'down {key}', item.shape)
            for key, item in up.items():
                print(f'up {key}', item.shape)
        return up


class RegistrationHead(nn.Sequential):
    """
    Registration head for generating displacement fields.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        # Initialize weights with small random values
        conv3d.weight = nn.Parameter(torch.zeros_like(conv3d.weight).normal_(0, 1e-5))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        self.add_module('conv3d', conv3d)


class HViT(nn.Module):
    """
    Hierarchical Vision Transformer (HViT) model for image registration.
    """
    def __init__(self, config: dict):
        super(HViT, self).__init__()
        self.upsample_df: bool = config.get('upsample_df', False)
        self.upsample_scale_factor: int = config.get('upsample_scale_factor', 2)
        self.scale_level_df: str = config.get('scale_level_df', 'P1')

        self.deformable: HierarchicalViT = HierarchicalViT(config)
        self.avg_pool: nn.AvgPool3d = nn.AvgPool3d(3, stride=2, padding=1)
        self.spatial_trans: SpatialTransformer = SpatialTransformer(config['data_size'])
        self.reg_head: RegistrationHead = RegistrationHead(
            in_channels=config.get('fpn_channels', 64),
            out_channels=ndims,
            kernel_size=ndims,
        )

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the HViT model.

        Args:
            source (Tensor): Source image tensor.
            target (Tensor): Target image tensor.

        Returns:
            Tuple[Tensor, Tensor]: Moved image and displacement field.
        """
        x: Tensor = torch.cat((source, target), dim=1)
        x_dec: Dict[str, Tensor] = self.deformable(x)

        # Extract features at the specified scale level
        x_dec: Tensor = x_dec[self.scale_level_df]
        flow: Tensor = self.reg_head(x_dec)

        if self.upsample_df:
            flow = nn.Upsample(scale_factor=self.upsample_scale_factor, 
                               mode='trilinear', 
                               align_corners=False)(flow)
        
        moved: Tensor = self.spatial_trans(source, flow)
        return moved, flow


if __name__ == "__main__":
    # Test the HViT model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    B = 1
    H, W, D = 160//2, 192//2, 224//2

    for fpn_channels in [64]:    
        config = {
            'NUM_CROSS_ATT': _NUM_CROSS_ATT,
            'out_fmaps': ['P4', 'P3', 'P2', 'P1'],
            'scale_level_df': 'P1',
            'upsample_df': True,
            'upsample_scale_factor': 2,
            'fpn_channels': fpn_channels,
            'start_channels': 32, 
            'patch_size': 2,
            'bspl': False,

            'backbone_net': 'fpn',
            'in_channels': 1,
            'data_size': [H, W, D],
            'bias': True,
            'norm_type': 'instance',
            'cuda': 0,
            'kernel_size': 3,
            'depths': 1, 
            'mlp_ratio': 2,

            'num_heads': 32, 
            'drop_path_rate': 0., 
            'qkv_bias': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,

            'use_seg_loss': False,
            'use_seg_proxy_loss': False,
            'num_organs': -1
        }

        source = torch.rand([1, 1, H, W, D]) 
        tgt = torch.rand([1, 1, H, W, D])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = HViT(config)
        model.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            source = source.to(dtype=torch.float16).to(device)
            tgt = tgt.to(dtype=torch.float16).to(device)

            moved, flow = model(source, tgt)
            print('\n\nmoved {} flow {}'.format(moved.shape, flow.shape))

            max_mem_mb = torch.cuda.max_memory_allocated() / 1024**3
            print("[+] Maximum memory:\t{:.2f}GB: >>> \t{:.0f} feats".format(max_mem_mb, config['fpn_channels']) if max_mem_mb is not None else "")
            print("[+] Required Total memory:\t{:.2f}GB".format(torch.cuda.get_device_properties(0).total_memory/1024**3))
            print("[+] Trainable params:\t{:.5f} m".format(count_parameters(model)/1e6))

