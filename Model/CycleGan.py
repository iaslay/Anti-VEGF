import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3), # no
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9, num_classes=3):
#         super(Generator, self).__init__()
#
#         # Initial convolution block
#         model_head = [nn.ReflectionPad2d(3),
#                       nn.Conv2d(input_nc, 64, 7),
#                       nn.InstanceNorm2d(64),
#                       nn.ReLU()]
#
#         # Downsampling
#         in_features = 64
#         out_features = in_features * 2
#         for _ in range(2):
#             model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                            nn.InstanceNorm2d(out_features),
#                            nn.ReLU()]
#             in_features = out_features
#             out_features = in_features * 2
#
#         # Residual blocks
#         model_body = []
#         for _ in range(n_residual_blocks):
#             model_body += [ResidualBlock(in_features)]
#
#         # Upsampling
#         model_tail = []
#         out_features = in_features // 2
#         for _ in range(2):
#             model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                            nn.InstanceNorm2d(out_features),
#                            nn.ReLU()]
#             in_features = out_features
#             out_features = in_features // 2
#
#         # Output layer
#         model_tail += [nn.ReflectionPad2d(3),
#                        nn.Conv2d(64, output_nc, 7),
#                        nn.Tanh()]
#
#         self.model_head = nn.Sequential(*model_head)
#         self.model_body = nn.Sequential(*model_body)
#         self.model_tail = nn.Sequential(*model_tail)
#         self.classifier_body = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1, 1),
#             nn.MaxPool2d(3, 2, 1),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.MaxPool2d(3, 2, 1)
#         )
#
#         self.classifier_tail = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes),
#         )
#         self.avgpool = nn.AvgPool2d(16, stride=1)
#
#     def forward(self, x):
#         x = self.model_head(x)
#         x = self.model_body(x) #[1, 256, 64, 64]
#         classifier_x = x.detach()
#         classifier_x = self.classifier_body(classifier_x)
#         classifier_x = self.avgpool(classifier_x)
#         classifier_x = classifier_x.view(classifier_x.size(0), -1)
#         classifier_x = self.classifier_tail(classifier_x)
#         x = self.model_tail(x)
#         # return x, classifier_x
#         return classifier_x
"""原始"""
# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(Generator, self).__init__()
#
#         # Initial convolution block
#         model_head = [nn.ReflectionPad2d(3),
#                       nn.Conv2d(input_nc, 64, 7),
#                       nn.InstanceNorm2d(64),
#                       nn.ReLU(inplace=True)]
#
#         # Downsampling
#         in_features = 64
#         out_features = in_features * 2
#         for _ in range(2):
#             model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                            nn.InstanceNorm2d(out_features),
#                            nn.ReLU(inplace=True)]
#             in_features = out_features
#             out_features = in_features * 2
#
#         # Residual blocks
#         model_body = []
#         for _ in range(n_residual_blocks):
#             model_body += [ResidualBlock(in_features)]
#
#         # Upsampling
#         model_tail = []
#         out_features = in_features // 2
#         for _ in range(2):
#             model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                            nn.InstanceNorm2d(out_features),
#                            nn.ReLU(inplace=True)]
#             in_features = out_features
#             out_features = in_features // 2
#
#         # Output layer
#         model_tail += [nn.ReflectionPad2d(3),
#                        nn.Conv2d(64, output_nc, 7),
#                        nn.Tanh()]
#
#         self.model_head = nn.Sequential(*model_head)
#         self.model_body = nn.Sequential(*model_body)
#         self.model_tail = nn.Sequential(*model_tail)
#
#     def forward(self, x):
#         x = self.model_head(x)
#         x = self.model_body(x)
#         x = self.model_tail(x)
#
#         return x

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=4, num_classes=3):
        super(Generator, self).__init__()
        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]
            model_body += [Block(dim=in_features, num_heads=1, mlp_ratio=2, qkv_bias=False, qk_scale=None, drop=0,
                                                 attn_drop=0, drop_path=0,norm_layer=torch.nn.LayerNorm, sr_ratio=8)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)
        self.classifier_body = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.MaxPool2d(2,2)
        )

        self.classifier_tail = nn.Sequential(
                nn.Linear(64*16*16, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, num_classes)
            )

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x) #[1, 256, 64, 64]
        classifier_x = x.contiguous().detach()
        classifier_x = self.classifier_body(classifier_x)
        classifier_x = classifier_x.view(classifier_x.size(0), -1)
        classifier_x = self.classifier_tail(classifier_x)
        x = self.model_tail(x)
        return x, classifier_x
#         return classifier_x

# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=6, num_classes=1):
#         super(Generator, self).__init__()
#         # Initial convolution block
#         model_head = [nn.ReflectionPad2d(3),
#                       nn.Conv2d(input_nc, 64, 7),
#                       nn.InstanceNorm2d(64),
#                       nn.ReLU(inplace=True)]
#
#         # Downsampling
#         in_features = 64
#         out_features = in_features * 2
#         self.model_head1 = nn.Sequential(CAPatchEmbedding(7,4, in_features,CAPE=True,
#                                             embed_dim=out_features),
#                            ParFormerBlock(dim=out_features, change=False),
#                                          ParFormerBlock(dim=out_features))
#
#         in_features = out_features
#         out_features = in_features * 2
#         self.model_head2 = nn.Sequential(CAPatchEmbedding(3,2 , in_features,CAPE=True,
#                                             embed_dim=out_features),
#                            ParFormerBlock(dim=out_features,change=False),
#                                          ParFormerBlock(dim=out_features))
#
#
#         in_features = out_features
#
#         # Residual blocks
#         model_body = []
#         for _ in range(n_residual_blocks):
#             model_body += [ResidualBlock(in_features)]
#             model_body += [ParFormerBlock(dim=in_features)]
#             model_body += [ParFormerBlock(dim=in_features)]
#
#         # Upsampling
#         self.model_head = nn.Sequential(*model_head)
#         self.model_body = nn.Sequential(*model_body)
#         # self.model_tail = nn.Sequential(*model_tail)
#
#         self.classifier_body1 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.MaxPool2d(2, 2),
#         )
#
#         self.classifier_body2 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1, 1),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 64, 3, 1, 1),
#         )
#
#         self.classifier_tail = nn.Sequential(
#             nn.Linear(64*16*16, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#             nn.Linear(2048, 256),
#             nn.Linear(256, 64),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.model_head(x)
#         x1 = self.model_head1(x)
#         x2 = self.model_head2(x1)
#         x = self.model_body(x2) #[1, 256, 64, 64]
#         classifier_x = x.detach()
#         classifier_x1 = x1.detach()
#         classifier_x = self.classifier_body2(classifier_x)
#         classifier_x1 = self.classifier_body1(classifier_x1)
#         classifier_x = classifier_x + classifier_x1
#         classifier_x = classifier_x.contiguous().view(classifier_x.size(0), -1)
#         classifier_x = self.classifier_tail(classifier_x)
#         return classifier_x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f'dim{dim} should be divided by num_heads{num_heads}'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = torch.nn.Linear(dim, dim * 2, bias=qkv_bias)
        # 将输入参数的张量随机设为0
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        # 对应paper里面的引入缩放率�?减少复杂�?        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = torch.nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # B num_head H*W C
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# class Attention(nn.Module):
#     """ Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
#     Modified from timm.
#     """
#     def __init__(self, dim, num_heads=None, head_dim=32, qk_scale=None, qkv_bias=True,
#                  attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
#         super().__init__()
#
#         self.head_dim = head_dim
#         self.scale = head_dim ** -0.5
#
#         self.num_heads = num_heads if num_heads else dim // head_dim
#         if self.num_heads == 0:
#             self.num_heads = 1
#
#         self.attention_dim = self.num_heads * self.head_dim
#
#         self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, H, W, C = x.shape
#         N = H * W
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q.transpose(-2, -1).contiguous() @ k) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (v @ attn).transpose(1, 2).contiguous().reshape(B, H, W, self.attention_dim)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2, act_layer=nn.GELU, bias=False, kernel_size=7, padding=3, **kwargs):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act = act_layer()

        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class ParFormerBlock(nn.Module):

    def __init__(self, dim, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 tokenmixer1=Attention, tokenmixer2=SepConv,
                 mlp=Mlp, mlp_ratio=4., layer_scale_init_value=1e-6,
                 drop=0., drop_path=0., shift=False,
                 block_num=0, change=True):
        super().__init__()

        cs1 = dim // 2
        cs2 = dim // 2
        self.split_index = (cs1, cs2)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.scale1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if shift and block_num % 2:
            self.shift = True
        else:
            self.shift = False

        if tokenmixer1 == tokenmixer2:
            self.tokenmixer1 = tokenmixer1(dim=dim, drop=drop)
            self.parallel = False
        else:
            self.tokenmixer1 = tokenmixer1(dim=cs1, drop=drop)
            self.tokenmixer2 = tokenmixer2(dim=cs2, drop=drop)
            self.parallel = True
        self.change = change

    def forward(self, x):
        if self.change:
            x = x.permute(0, 2, 3, 1)
        _, _, _, C = x.shape
        x = self.norm1(x)
        if self.parallel:
            x1, x2 = torch.split(x, self.split_index, dim=3)
            x1 = self.tokenmixer1(x1)
            x2 = self.tokenmixer2(x2)
            if self.shift:
                xs = torch.cat((x2, x1), dim=3)  # channel join
            else:
                xs = torch.cat((x1, x2), dim=3)  # channel join
        else:
            xs = self.tokenmixer1(x)

        x = x + self.scale1 * self.drop_path1(xs)
        x = x + self.scale2 * self.drop_path2(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)

        return x

class CAPatchEmbedding(nn.Module):
    """ Channel Attention Pacth Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, act_layer=nn.GELU, CAPE=False, embed_dim=768):
        super().__init__()
        conv_kernel  = to_2tuple(patch_size)
        pool_kernel  = to_2tuple(patch_size+(patch_size//2))
        pool_padding = to_2tuple(patch_size//2)

        if patch_size > 3:
            conv_padding = to_2tuple(patch_size//2-1)
        else:
            conv_padding = to_2tuple(patch_size//2)

        self.conv_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_kernel, stride=stride,
                              padding=conv_padding)
        if CAPE:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=stride,
                                padding=pool_padding)
            self.pw_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
            self.act = act_layer()
            self.cape = True
        else:
            self.cape = False

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        if self.cape:
            xr = self.act(self.pw_proj(self.pool(x))) # Channel Attention
            x  = self.conv_proj(x) # Convolutional Patch Embedding
            x  = x + xr
        else:
            x  = self.conv_proj(x) # Convolutional Patch Embedding
        x  = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x  = self.norm(x)
        return x




