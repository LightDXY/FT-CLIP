from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

global LAYER_NORM 
LAYER_NORM = True


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if LAYER_NORM:
            ret = super().forward(x)
        else:
            ret = x
        return ret
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0., drop_path: float=0, mlp_width: int = 0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        mlp_w = mlp_width or d_model * 4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_w)),
            ("gelu", QuickGELU()),
            # ("dropout_1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(mlp_w, d_model)),
            # ("dropout_2", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                    checkpoint: bool = False, dropout: float = 0., mlp_width: int=0,
                    emb_dropout: float = 0., drop_path_rate: float=0,
                    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.checkpoint = checkpoint
        self.dropout = nn.Dropout(emb_dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        print(f"Using DPR {dpr}")
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, mlp_width=mlp_width,
                                     dropout=dropout, drop_path=dpr[i],
                                     ) for i in range(layers)])



    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input = input.detach()
            input.requires_grad = True
        return checkpoint_sequential(layer, segments, input)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        if self.checkpoint:
            return self.checkpoint_fwd(self.resblocks, x, self.layers)

        x = self.resblocks(x)
        return x

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int,
                    heads: int, checkpoint: bool, dropout: float=0, emb_dropout: float=0,
                    drop_path_rate: float=0, mlp_width: int = 0,
                    freeze_conv1: bool = True):
        super().__init__()
        self.input_resolution = input_resolution
        self.freeze_conv1 = freeze_conv1

        self.patch_embed = PatchEmbed(
            img_size=input_resolution, patch_size=patch_size, in_chans=3, embed_dim=width)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.layers = layers
        self.transformer = Transformer(width, layers, heads, checkpoint=checkpoint, 
                                        dropout=dropout, emb_dropout=emb_dropout, mlp_width = mlp_width,
                                        drop_path_rate=drop_path_rate)

        #self.ln_post = LayerNorm(width)
        #self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.initialize_parameters()

        init_scale = 0.001
        self.fc_norm = LayerNorm(width)
        self.head = nn.Linear(width, 1000)
        trunc_normal_(self.head.weight, std=.02)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)

        self.train()
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def get_num_layers(self):
        return self.layers

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        f_list = [self.patch_embed] ### provide the patch and seq_len to the main.py, not used in the training.
        if self.freeze_conv1: ### freeze the conv1, copied from the DeCLIP codebase and we followed its default setting.
            f_list.append(self.conv1)

        print('-----------------------------------------------------------')

        for layer in f_list:
            print(f"set {layer}.requires_grad to False")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
        return self


    def forward(self, x: torch.Tensor, return_dense=False, return_feature=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.fc_norm(x[:,1:,:])
        x = self.head(x.mean(1))
        return x

@register_model
def CLIP_B16(pretrained=False, **kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64
    print (kwargs)

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads, 
        'input_resolution': 224,
        'patch_size': 16,
        'width': vision_width,
        'checkpoint': False,
    }
    model = VisualTransformer(**default_kwargs)
    model.train()
    return model

@register_model
def CLIP_B16_384(pretrained=False, **kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64
    print (kwargs)

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads, 
        'input_resolution': 384,
        'patch_size': 16,
        'width': vision_width,
        'checkpoint': False,
        'freeze_conv1': False,
    }
    model = VisualTransformer(**default_kwargs)
    model.train()
    return model

@register_model
def CLIP_L14(pretrained=False, **kwargs):
    vision_width = 1024
    vision_layers = 24
    vision_heads = vision_width // 64
    print (kwargs)

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads, 
        'input_resolution': 224,
        'patch_size': 14,
        'width': vision_width,
        'checkpoint': False,
    }
    model = VisualTransformer(**default_kwargs)
    model.train()
    return model

@register_model
def CLIP_L14_336(pretrained=False, **kwargs):
    vision_width = 1024
    vision_layers = 24
    vision_heads = vision_width // 64
    print (kwargs)

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads, 
        'input_resolution': 336,
        'patch_size': 14,
        'width': vision_width,
        'checkpoint': False,
        'freeze_conv1': False,
    }
    model = VisualTransformer(**default_kwargs)
    model.train()
    return model


