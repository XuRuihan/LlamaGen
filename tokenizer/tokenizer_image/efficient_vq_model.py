# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    codebook_heads: int = 1
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0



class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage, heads=config.codebook_heads)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Conv2d, nn.Linear)):
    #         nn.init.trunc_normal_(module.weight.data, std=0.02)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias.data)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff



class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch * ch_mult[0], kernel_size=3, stride=1, padding=1)

        # downsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                block_out = ch*ch_mult[i_level+1]
                conv_block.downsample = Downsample(block_in, block_out, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                block_out = ch*ch_mult[i_level-1]
                conv_block.upsample = Upsample(block_in, block_out, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, heads):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.heads = heads

        self.embedding = nn.Parameter(torch.zeros(heads, n_e, e_dim // heads))
        self.embedding.data.normal_(std=e_dim ** -0.5)
        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z: torch.Tensor):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b (d e) h w -> d b h w e", d=self.heads).contiguous()
        z_flattened = z.detach().view(self.heads, -1, self.e_dim // self.heads)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding, p=2, dim=-1)
        else:
            embedding = self.embedding

        d = torch.sum(z_flattened ** 2, dim=-1).unsqueeze(-1) + \
            torch.sum(embedding ** 2, dim=-1).unsqueeze(-2) - 2 * \
            z_flattened @ embedding.mT

        min_encoding_indices = torch.argmin(d, dim=-1)
        z_q = embedding[torch.arange(self.heads, device=z.device).unsqueeze(-1), min_encoding_indices]
        z_q = z_q.view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            indices = (min_encoding_indices + torch.arange(0, self.n_e * self.heads, step=self.n_e, device=z.device).unsqueeze(-1)).flatten()
            cur_len = indices.shape[0]
            self.codebook_used = torch.cat((self.codebook_used[cur_len:], indices))
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e / self.heads

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            entropy_loss = self.entropy_loss_ratio * self.compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'd b h w e -> b (d e) h w')
        min_encoding_indices = rearrange(min_encoding_indices, 'd (b h w) -> b d h w', b=z_q.shape[0], h=z_q.shape[2], w=z_q.shape[3])

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding, p=2, dim=-1)
        else:
            embedding = self.embedding

        B, D, H, W = indices.shape
        indices = rearrange(indices, 'b d h w -> d (b h w)')
        z_q = embedding[torch.arange(self.heads, device=embedding.device).unsqueeze(-1), indices]
        if channel_first:
            z_q = rearrange(z_q, 'd (b h w) e -> b (d e) h w', b=B, h=H, w=W)
        else:
            z_q = rearrange(z_q, 'd (b h w) e -> b h w (d e)', b=B, h=H, w=W)

        return z_q

    @staticmethod
    def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])
        flat_affinity /= temperature
        probs = F.softmax(flat_affinity, dim=-1)
        log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
        if loss_type == "softmax":
            target_probs = probs
        else:
            raise ValueError("Entropy loss {} not supported".format(loss_type))
        avg_probs = torch.mean(target_probs, dim=0)
        avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
        loss = sample_entropy - avg_entropy
        return loss


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels == in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = Normalize(in_channels, norm_type)
        self.pwconv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        self.pwconv2 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = x
        h = self.dwconv(h)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.pwconv2(h)
        return x+h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q, k, v = self.qkv(h_).chunk(3, dim=1)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    return F.silu(x)
    # return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        assert with_conv is True
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        assert with_conv is True
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
