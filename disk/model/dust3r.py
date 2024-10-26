# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy

from torch_dimcheck import dimchecked

inf = float('inf')
import os

import torch
from torch import nn
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import load_model
from dust3r.utils.misc import transpose_to_landscape, fill_default_args, freeze_all_params  # noqa
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401

from models.head_downstream import PixelwiseTaskWithDPT
from models.croco import CroCoNet  # noqa


class LinearHead(nn.Module):
    """
    Linear head
    Each token outputs: - 16x16 desc size + 1 confidence
    """

    def __init__(self, net, num_outputs: int):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode

        self.proj = nn.Linear(net.dec_embed_dim, num_outputs * self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        return feat

class DUSt3R(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self,
                 output_mode='desc',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 desc_dim=128,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)
        self.desc_dim = desc_dim
        # dust3r specific initialization
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            # model = CroCoNet(**ckpt.get('croco_kwargs', {})).to(device)
            # model.eval()
            # msg = model.load_state_dict(ckpt['model'], strict=True)
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(DUSt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        if self.head_type == 'linear':
            self.downstream_head1 = LinearHead(self, self.desc_dim + bool(self.conf_mode))
        elif self.head_type == 'dpt':
            """
                return PixelwiseTaskWithDPT for given net params
                """
            assert self.dec_depth > 9
            l2 = self.dec_depth
            feature_dim = 256
            last_dim = feature_dim // 2
            out_nchan = self.desc_dim
            ed = self.enc_embed_dim
            dd = self.dec_embed_dim
            self.downstream_head1 = PixelwiseTaskWithDPT(num_channels=out_nchan + 1,
                                        feature_dim=feature_dim,
                                        last_dim=last_dim,
                                        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                        dim_tokens=[ed, dd, dd, dd],
                                        postprocess=None,
                                        depth_mode=self.depth_mode,
                                        conf_mode=self.conf_mode,
                                        head_type='regression')
            self.downstream_head1.setup(self)
        else:
            raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img2 is None:
            out, pos, _ = self._encode_image(img1, true_shape1)
            return out, out, pos, pos
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2


    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1 in self.dec_blocks:
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk1(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    @dimchecked
    def forward(self, img1: ['B', 'C', 'H', 'W'], shape1=None, img2=None, shape2=None):
        # encode the two images --> B,S,D
        B = img1.shape[0]
        shape1 = shape1 if shape1 is not None else  torch.tensor(img1.shape[-2:])[None].repeat(B, 1)

        # encode img1 only if img2 is None and return output twice
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            feat = self._downstream_head(1, [tok.float() for tok in dec1], shape1)

        return feat


if __name__ == '__main__':
    model = DUSt3R(img_size=224,  # input image size
                   patch_size=16,  # patch_size
                   enc_embed_dim=1024,  # encoder feature dimension
                   enc_depth=24,  # encoder depth
                   enc_num_heads=12,  # encoder number of heads in the transformer block
                   dec_embed_dim=512,  # decoder feature dimension
                   dec_depth=8,  # decoder depth
                   dec_num_heads=16,  # decoder number of heads in the transformer block
                   mlp_ratio=4,
                   norm_im2_in_dec=True,
                   # whether to apply normalization of the 'memory' = (second image) in the decoder
                   pos_embed='RoPE100',  # positional embedding (either cosine or RoPE100)
                   )

    input = torch.rand((1, 3, 224, 224))

    with torch.no_grad():
        output = model.forward(input, input)  # self attention
    print(output)