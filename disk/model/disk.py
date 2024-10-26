import os

import torch
import numpy as np

from torch_dimcheck import dimchecked
from unets import Unet, thin_setup

from disk import NpArray, Features
from disk.model.detector import Detector

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}

class DISK(torch.nn.Module):
    def __init__(
        self,
        args,
        setup=DEFAULT_SETUP,
        kernel_size=5,
    ):
        super(DISK, self).__init__()

        self.desc_dim = args.desc_dim
        self.backbone = args.backbone
        if self.backbone == "unet":
            self.model = Unet(
                in_features=3, size=kernel_size,
                down=[16, 32, 64, 64, 64],
                up=[64, 64, 64, self.desc_dim+1],
                setup=setup,
            )
        elif self.backbone == "dust3r":
            from disk.model.dust3r import DUSt3R
            self.model = DUSt3R(pos_embed='RoPE100',
                                img_size=(224, 224),
                                head_type='dpt',
                                freeze=self.freeze,
                                enc_embed_dim=1024,
                                enc_depth=24,
                                enc_num_heads=16,
                                dec_embed_dim=768,
                                dec_depth=12,
                                dec_num_heads=12,
                                landscape_only=False,
                                desc_dim=self.desc_dim)  # positional embedding (either cosine or RoPE100))
            taskId = int(os.getenv('SLURM_ARRAY_JOB_ID', 0))
            if taskId == 0:
                ckpt = torch.load("CroCo_V2_ViTLarge_BaseDecoder.pth", map_location='cpu')
                s = self.model.load_state_dict(ckpt['model'], strict=False)
                print("Croco weights loaded", s)
        elif self.backbone == "mickey":
            from disk.model.mickey import MicKey
            self.model = MicKey()
        else:
            raise ValueError("backbone type not implemented")

        self.detector = Detector(window=window)

    @dimchecked
    def _split(self, unet_output: ['B', 'C', 'H', 'W']) \
                -> (['B', 'C-1', 'H', 'W'], ['B', 1, 'H', 'W']):
        '''
        Splits the raw Unet output into descriptors and detection heatmap.
        '''
        assert unet_output.shape[1] == self.desc_dim + 1

        descriptors = unet_output[:, :self.desc_dim]
        heatmap     = unet_output[:, self.desc_dim:]

        return descriptors, heatmap

    @dimchecked
    def features(
        self,
        images: ['B', 'C', 'H', 'W'],
        true_shapes: ['B', '2'],
        kind='rng',
        **kwargs
    ) -> (..., ['B', '1', 'H', 'W']):
        '''
            true_shape: real image shape before resizing/padding
            allowed values for `kind`:
            * rng
            * nms
        '''

        B = images.shape[0]
        try:
            if self.backbone in ["unet", "dust3r"]:
                model_output = self.model(images)
            else:
                model_output = self.model(images, true_shapes)
            descriptors, heatmaps = self._split(model_output)
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = ('U-Net failed because the input is of wrong shape. With '
                       'a n-step U-Net (n == 4 by default), input images have '
                       'to have height and width as multiples of 2^n (16 by '
                       'default).')
                raise RuntimeError(msg) from e
            else:
                raise

        keypoints = {
            'rng': self.detector.sample,
            'nms': self.detector.nms,
        }[kind](heatmaps, **kwargs)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return np.array(features, dtype=object), heatmaps

    # @dimchecked
    # def features_multi(
    #     self,
    #     images1: ['B', 'C', 'H', 'W'],
    #     images2: ['B', 'C', 'H', 'W'],
    #     true_shapes1,
    #     true_shapes2,
    #     kind='rng',
    #     **kwargs
    # ) -> NpArray[Features]:
    #     '''
    #         true_shape: real image shape before resizing/padding
    #         allowed values for `kind`:
    #         * rng
    #         * nms
    #     '''
    #
    #     B = images.shape[0]
    #     try:
    #         if self.backbone in ["unet"]:
    #             model_output = self.model(images)
    #         else:
    #             model_output = self.model(images, true_shapes)
    #         descriptors, heatmaps = self._split(model_output)
    #     except RuntimeError as e:
    #         if 'Trying to downsample' in str(e):
    #             msg = ('U-Net failed because the input is of wrong shape. With '
    #                    'a n-step U-Net (n == 4 by default), input images have '
    #                    'to have height and width as multiples of 2^n (16 by '
    #                    'default).')
    #             raise RuntimeError(msg) from e
    #         else:
    #             raise
    #
    #     keypoints = {
    #         'rng': self.detector.sample,
    #         'nms': self.detector.nms,
    #     }[kind](heatmaps, **kwargs)
    #
    #     features = []
    #     for i in range(B):
    #         features.append(keypoints[i].merge_with_descriptors(descriptors[i]))
    #
    #     return np.array(features, dtype=object), heatmaps
